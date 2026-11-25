"""
Fine-tunes a pretrained transformer (XLM-R or SlavicBERT) for word-level language ID in
one of the training settings from splits.py, and evaluates it on every evaluation set,
like train_crf.py and train_bilstm.py. The models read the lowercased Cyrillic column.

Early stopping: after every epoch the model labels the development family (fam_sh);
the weights of the best epoch are kept, and training stops after PATIENCE epochs
without improvement.

Meant for a GPU (Google Colab). On a GPU, training uses mixed precision (float16), which is
about twice as fast; on CPU everything runs in float32 and is very slow for the full models.

Run from the project root:
    python src/transformer/train_transformer.py --model xlmr                         # zero-shot, results/xlmr.json
    python src/transformer/train_transformer.py --model xlmr --setting in_domain     # results/xlmr_in_domain.json
    python src/transformer/train_transformer.py --model slavicbert --all             # every run in ALL_RUNS
    python src/transformer/train_transformer.py --model xlmr --setting mixed --limit 200 --max-epochs 1
                                                     # quick test on 200 sentences per set; saves nothing
"""
import sys
import time
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.metrics import evaluate_predictions
from src.baseline.splits import (training_sentences, evaluation_sets, evaluate_on_sets, trasyanka_split,
                                 print_macro_f1_table, save_results, save_predictions,
                                 TRAINING_SETTINGS, LEARNING_CURVE_FRACTIONS, RANDOM_SEED)
from src.model.data import LABEL_TO_ID, ID_TO_LABEL
from src.model.train_bilstm import set_seed
from src.transformer.data import MODEL_NAMES, FIELD, load_tokenizer, make_loader, word_predictions

BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 3e-5      # the usual fine-tuning range is 2e-5 to 5e-5
WEIGHT_DECAY = 0.01
WARMUP_SHARE = 0.1        # the learning rate rises from 0 over the first 10 % of steps, then falls linearly
MAX_EPOCHS = 5            # pretrained models converge in a few epochs
PATIENCE = 2
GRAD_CLIP = 1.0

ALL_RUNS = ([("zero_shot", 1.0), ("in_domain", 1.0), ("mixed", 1.0)]
            + [("mixed", fraction) for fraction in LEARNING_CURVE_FRACTIONS])


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_name(model_key, setting, fraction):
    """xlmr, xlmr_in_domain, slavicbert_mixed_30 ... (same scheme as the CRF and BiLSTM)"""
    if setting == "zero_shot":
        return model_key
    name = f"{model_key}_{setting}"
    if fraction != 1.0:
        name += f"_{round(fraction * 100)}"
    return name


def load_model(model_key, device):
    """The pretrained encoder with a new, randomly initialized 4-label classification layer on top."""
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAMES[model_key], num_labels=len(LABEL_TO_ID), id2label=ID_TO_LABEL, label2id=LABEL_TO_ID)
    return model.to(device)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device):
    """
    One pass over the training loader; returns the mean loss over its batches.

    The model computes the loss itself when it gets labels: cross-entropy over the pieces whose
    label is not PAD_LABEL (-100), i.e. each word's first piece. On a GPU the forward pass runs in
    float16 and the GradScaler scales the loss so small gradients don't round to zero; on CPU the
    scaler is disabled and the same calls do a plain float32 step.
    Arguments go to the model by keyword: its third positional parameter is token_type_ids, not labels.
    """
    total_loss = 0.0
    model.train()           # dropout on
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)              # back to the real gradient sizes, so clipping sees them
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)                  # skipped if the gradients overflowed in float16
        scaler.update()
        scheduler.step()                        # warmup / linear decay, once per batch
        total_loss += loss.item()
    return total_loss / len(loader)


def predict(model, loader, device):
    """
    Predicted word label lists for every sentence in the loader, in loader order, one label per
    word. The loader must not shuffle.
    """
    predictions = []
    model.eval()            # dropout off
    with torch.no_grad():   # no gradients needed: saves memory and time
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits   # [batch, pieces, 4]
            piece_label_ids = logits.argmax(dim=-1)                                       # [batch, pieces]
            for b in range(len(piece_label_ids)):
                # .tolist(): plain ints for ID_TO_LABEL; word_ids is already a list
                predictions.append(word_predictions(piece_label_ids[b].tolist(), batch["word_ids"][b]))
    return predictions


def predict_words(model, tokenizer, word_lists, device):
    """
    Label lists for plain Cyrillic word lists, the interface evaluate_on_sets expects.
    Every word gets a placeholder label, which the dataset needs and predict() never looks at.
    """
    records = [{FIELD: words, "labels": ["SHARED"] * len(words)} for words in word_lists]
    return predict(model, make_loader(records, tokenizer, batch_size=EVAL_BATCH_SIZE), device)


def dev_macro_f1(model, dev_loader, dev_records, device):
    """Macro-F1 over all four labels on the development family."""
    predictions = predict(model, dev_loader, device)
    return evaluate_predictions([record["labels"] for record in dev_records], predictions)["macro_f1"]


def train_and_evaluate(model_key, setting, fraction, sets, max_epochs=MAX_EPOCHS, patience=PATIENCE, limit=None):
    """
    Fine-tunes one model with early stopping on fam_sh and scores the best epoch's weights on `sets`.
    Returns (results, history). limit (for quick tests) keeps only the first `limit` sentences
    of the training data, of fam_sh and of every evaluation set.
    As for the CRF and BiLSTM, in_domain and mixed runs skip the full "trasyanka" set.
    """
    set_seed()
    device = get_device()
    tokenizer = load_tokenizer(model_key)
    train = training_sentences(setting, fraction)
    dev = trasyanka_split("dev")
    if limit:
        train, dev = train[:limit], dev[:limit]
        sets = {name: records[:limit] for name, records in sets.items()}
    if setting != "zero_shot":
        sets = {name: records for name, records in sets.items() if name != "trasyanka"}

    train_loader = make_loader(train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = make_loader(dev, tokenizer, batch_size=EVAL_BATCH_SIZE)

    model = load_model(model_key, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = max_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_SHARE * total_steps), total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    print(f"{len(train):,} training sentences, {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"device {device}, mixed precision {scaler.is_enabled()}")

    history, best_f1, best_state, epochs_without_improvement = [], -1, None, 0
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        f1 = dev_macro_f1(model, dev_loader, dev, device)
        history.append(f1)
        if f1 > best_f1:
            best_f1, epochs_without_improvement = f1, 0
            # A copy on the CPU: GPU memory is needed for training, and the weights keep changing
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            marker = "  best"
        else:
            epochs_without_improvement += 1
            marker = ""
        print(f"epoch {epoch}  loss {loss:.4f}  dev Macro-F1 {f1:.4f}  {(time.time() - start) / 60:.1f} min{marker}",
              flush=True)
        if epochs_without_improvement >= patience:
            print(f"No improvement for {patience} epochs, stopping")
            break

    model.load_state_dict(best_state)
    results = evaluate_on_sets(lambda word_lists: predict_words(model, tokenizer, word_lists, device), sets,
                               field=FIELD)
    return results, history


def main():
    """Parses the arguments, runs each requested (setting, fraction) and saves results/<run_name>.json."""
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate a transformer.")
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), required=True)
    parser.add_argument("--setting", choices=TRAINING_SETTINGS, default="zero_shot")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="share of the Trasyanka training transcripts (in_domain / mixed only)")
    parser.add_argument("--all", action="store_true", help="run every (setting, fraction) in ALL_RUNS")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--limit", type=int, help="quick test: only this many sentences per set; saves nothing")
    args = parser.parse_args()

    runs = ALL_RUNS if args.all else [(args.setting, args.fraction)]
    sets = evaluation_sets()

    for setting, fraction in runs:
        name = run_name(args.model, setting, fraction)
        print(f"\n=== {name}: setting={setting}, fraction={fraction} ===")
        start = time.time()
        results, history = train_and_evaluate(args.model, setting, fraction, sets,
                                              max_epochs=args.max_epochs, limit=args.limit)
        best_epoch = history.index(max(history)) + 1
        print(f"Best epoch {best_epoch} of {len(history)} (dev Macro-F1 {max(history):.4f}), "
              f"{(time.time() - start) / 60:.1f} min")
        print_macro_f1_table(name, results)
        print(results["trasyanka_test"]["report"])
        if args.limit:
            print("--limit: quick test, results not saved")
            continue
        save_results(name, results)
        save_predictions(name, sets, results)


if __name__ == "__main__":
    main()
