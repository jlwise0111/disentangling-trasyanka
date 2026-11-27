"""
Trains the hierarchical BiLSTM in one of the training settings from splits.py
(zero_shot, in_domain, mixed) and evaluates it on every evaluation set, like train_crf.py.

Early stopping: after every epoch the model labels the development family (fam_sh).
The weights from the epoch with the best development Macro-F1 are kept, and training
stops after PATIENCE epochs without improvement.

Run from the project root with the .venv Python:
    python src/model/train_bilstm.py                                  # zero-shot, results/bilstm.json
    python src/model/train_bilstm.py --setting in_domain              # results/bilstm_in_domain.json
    python src/model/train_bilstm.py --setting mixed --fraction 0.3   # results/bilstm_mixed_30.json
    python src/model/train_bilstm.py --all                            # every run in ALL_RUNS
"""
import sys
import copy
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.metrics import evaluate_predictions
from src.baseline.splits import (training_sentences, evaluation_sets, evaluate_on_sets, trasyanka_split,
                                 print_macro_f1_table, save_results, save_predictions,
                                 TRAINING_SETTINGS, LEARNING_CURVE_FRACTIONS, RANDOM_SEED)
from src.model.data import build_char_vocab, make_loader, ID_TO_LABEL, PAD_LABEL
from src.model.bilstm import HierarchicalBiLSTM, NUM_LABELS, count_parameters

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 20
PATIENCE = 3              # epochs without a better development Macro-F1 before stopping
GRAD_CLIP = 5.0           # largest allowed gradient norm; keeps LSTM training stable

# Same runs as the CRF, so the two models can be compared setting by setting
ALL_RUNS = ([("zero_shot", 1.0), ("in_domain", 1.0), ("mixed", 1.0)]
            + [("mixed", fraction) for fraction in LEARNING_CURVE_FRACTIONS])


def set_seed(seed=RANDOM_SEED):
    """Same seed -> same weights, same shuffling, same results (on the same machine)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_name(setting, fraction):
    """bilstm, bilstm_in_domain, bilstm_mixed, bilstm_mixed_30 ... (same scheme as train_crf.run_name)"""
    if setting == "zero_shot":
        return "bilstm"
    name = f"bilstm_{setting}"
    if fraction != 1.0:
        name += f"_{round(fraction * 100)}"
    return name


def train_one_epoch(model, loader, optimizer, loss_fn):
    """
    One pass over the training loader. Returns the mean loss over its batches.
    """
    total_loss = 0.0
    model.train()           # dropout on
    for chars, labels, word_mask in loader:
        optimizer.zero_grad()
        scores = model(chars, word_mask)
        # CrossEntropyLoss wants [N, classes] and [N]; positions labeled PAD_LABEL are skipped
        loss = loss_fn(scores.reshape(-1, NUM_LABELS), labels.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def predict(model, loader):
    """
    Predicted label strings for every sentence in the loader, in loader order, each list
    as long as its sentence (no padding). The loader must not shuffle.
    """
    model.eval()            # dropout off
    predictions = []
    with torch.no_grad():   # no gradients needed for prediction
        for chars, labels, word_mask in loader:
            scores = model(chars, word_mask)
            ids = scores.argmax(dim=-1)                 # [batch, sentence]
            for b in range(ids.shape[0]):
                real_ids = ids[b][word_mask[b]]         # only the real words, in order
                predictions.append([ID_TO_LABEL[i.item()] for i in real_ids])

    return predictions


def predict_tokens(model, token_lists, char_vocab):
    """
    Label lists for plain token lists, the interface evaluate_on_sets expects.
    The dataset needs labels to build batches, so every word gets a placeholder label
    that predict() never looks at.
    """
    records = [{"tokens": tokens, "labels": ["SHARED"] * len(tokens)} for tokens in token_lists]
    return predict(model, make_loader(records, char_vocab, batch_size=256))


def dev_macro_f1(model, dev_loader, dev_records):
    """Macro-F1 over all four labels on the development family."""
    predictions = predict(model, dev_loader)
    gold = [record["labels"] for record in dev_records]
    return evaluate_predictions(gold, predictions)["macro_f1"]


def train_and_evaluate(setting, fraction, sets, seed=RANDOM_SEED, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """
    Trains one model with early stopping on fam_sh and scores the best epoch's weights on `sets`.
    Returns (results, history, model):
        results  {set name: metrics}, as evaluate_on_sets returns
        history  development Macro-F1 after each epoch that ran
        model    the model with the best epoch's weights loaded
    As for the CRF, in_domain and mixed runs skip the full "trasyanka" set (it contains their training families).
    """
    set_seed(seed)
    train = training_sentences(setting, fraction, seed)
    dev = trasyanka_split("dev")
    char_vocab = build_char_vocab(train)        # from this run's training data only

    train_loader = make_loader(train, char_vocab, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = make_loader(dev, char_vocab, batch_size=256)

    model = HierarchicalBiLSTM(len(char_vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    print(f"{len(train):,} training sentences, {len(char_vocab)} characters, "
          f"{count_parameters(model):,} parameters")

    history = []
    best_f1 = -1
    best_state = None
    epochs_wo_improvement = 0
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        f1 = dev_macro_f1(model, dev_loader, dev)
        history.append(f1)

        if f1 > best_f1:            # a tie is not an improvement
            best_f1 = f1
            # deepcopy: state_dict() returns the live tensors, which keep training
            best_state = copy.deepcopy(model.state_dict())
            epochs_wo_improvement = 0
            marker = "  best"
        else:
            epochs_wo_improvement += 1
            marker = ""
        print(f"epoch {epoch:>2}  loss {loss:.4f}  dev Macro-F1 {f1:.4f}  {time.time() - start:.0f}s{marker}",
              flush=True)

        if epochs_wo_improvement >= patience:
            print(f"No improvement for {patience} epochs, stopping")
            break

    model.load_state_dict(best_state)

    if setting != "zero_shot":
        sets = {name: records for name, records in sets.items() if name != "trasyanka"}
    results = evaluate_on_sets(lambda token_lists: predict_tokens(model, token_lists, char_vocab), sets)
    return results, history, model


def main():
    """Parses the arguments, runs each requested (setting, fraction) and saves results/<run_name>.json."""
    parser = argparse.ArgumentParser(description="Train and evaluate the hierarchical BiLSTM.")
    parser.add_argument("--setting", choices=TRAINING_SETTINGS, default="zero_shot")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="share of the Trasyanka training transcripts (in_domain / mixed only)")
    parser.add_argument("--all", action="store_true", help="run every (setting, fraction) in ALL_RUNS")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    runs = ALL_RUNS if args.all else [(args.setting, args.fraction)]
    sets = evaluation_sets()        # loaded once, shared by every run

    for setting, fraction in runs:
        name = run_name(setting, fraction)
        print(f"\n=== {name}: setting={setting}, fraction={fraction} ===")
        start = time.time()
        results, history, _ = train_and_evaluate(setting, fraction, sets, max_epochs=args.max_epochs)
        best_epoch = history.index(max(history)) + 1
        print(f"Best epoch {best_epoch} of {len(history)} (dev Macro-F1 {max(history):.4f}), "
              f"{(time.time() - start) / 60:.1f} min")
        print_macro_f1_table(name, results)
        print(results["trasyanka_test"]["report"])
        save_results(name, results)
        save_predictions(name, sets, results)


if __name__ == "__main__":
    main()
