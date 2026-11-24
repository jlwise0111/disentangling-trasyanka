"""
Tunes the CRF's regularisation (c1 = L1 penalty, c2 = L2 penalty) on the development
family fam_sh, so the CRF gets the same chance as the BiLSTM, which picks its best
epoch on fam_sh. The test families are never looked at here.

For each training setting (fraction 1.0) every (c1, c2) pair in GRID is trained and
scored on fam_sh. The scores and the best pair per setting go to results/crf_tuning.json;
train_crf.py can then retrain with the best pair and score the test families once.

Run from the project root:
    python src/baseline/tune_crf.py                       # all three settings (about 30-45 minutes)
    python src/baseline/tune_crf.py --setting in_domain   # one setting
"""
import sys
import json
import time
import argparse
from itertools import product
from pathlib import Path

import sklearn_crfsuite

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.conllu_parser import load_both_vocabs
from src.utils.metrics import evaluate_predictions
from src.baseline.crf_features import sent_to_features
from src.baseline.splits import training_sentences, trasyanka_split, TRAINING_SETTINGS, RESULTS_DIR
from src.baseline.train_crf import CRF_PARAMS

# Ten times apart, so the grid spans weak to strong regularization; 0.1 / 0.1 is the current default
C1_VALUES = (0.01, 0.1, 1.0)
C2_VALUES = (0.01, 0.1, 1.0)
GRID = list(product(C1_VALUES, C2_VALUES))      # 9 (c1, c2) pairs
TUNING_PATH = RESULTS_DIR / "crf_tuning.json"


def features(records, ru_vocab, be_vocab):
    """
    CRF input for a list of records: (X, y), where X holds sent_to_features(...) of every
    record's tokens and y every record's labels, in the same order.
    """
    X = [sent_to_features(record["tokens"], ru_vocab, be_vocab) for record in records]
    y = [record["labels"] for record in records]

    return X, y


def dev_macro_f1(crf, X_dev, dev_records):
    """Macro-F1 over all four labels of a trained CRF on the development family."""
    gold = [record["labels"] for record in dev_records]
    predictions = crf.predict(X_dev)
    scores = evaluate_predictions(gold, predictions)
    return scores["macro_f1"]


def tune_setting(setting, ru_vocab, be_vocab, dev_records, grid=GRID):
    """
    Trains one CRF per (c1, c2) pair in grid on training_sentences(setting) and scores it on
    dev_records. Returns
        {"scores": [{"c1": c1, "c2": c2, "dev_macro_f1": f1}, ...],   # one per pair, in grid order
         "best":   {"c1": c1, "c2": c2, "dev_macro_f1": f1}}           # the highest score
    """
    X_train, y_train = features(training_sentences(setting), ru_vocab, be_vocab)
    X_dev, _ = features(dev_records, ru_vocab, be_vocab)

    scores = []
    best = {"c1": None, "c2": None, "dev_macro_f1": -1}
    for c1, c2 in grid:
        start = time.time()
        crf = sklearn_crfsuite.CRF(**{**CRF_PARAMS, "c1": c1, "c2": c2})
        crf.fit(X_train, y_train)
        f1 = dev_macro_f1(crf, X_dev, dev_records)
        scores.append({"c1": c1, "c2": c2, "dev_macro_f1": f1})
        duration = time.time() - start
        print(f"c1={c1:<5} c2={c2:<5} dev Macro-F1 {f1:.4f}  {duration:.0f}s", flush=True)
        if f1 > best["dev_macro_f1"]:       # a tie keeps the earlier, weaker-regularised pair
            best = {"c1": c1, "c2": c2, "dev_macro_f1": f1}

    return {"scores": scores, "best": best}


def main():
    """Tunes every requested setting and saves the scores and best pairs to results/crf_tuning.json."""
    parser = argparse.ArgumentParser(description="Tune the CRF's c1 / c2 on the development family.")
    parser.add_argument("--setting", choices=TRAINING_SETTINGS, help="only this setting (default: all)")
    args = parser.parse_args()
    settings = [args.setting] if args.setting else list(TRAINING_SETTINGS)

    ru_vocab, be_vocab = load_both_vocabs()
    dev = trasyanka_split("dev")

    # Keep earlier settings' results when only one setting is rerun
    tuning = {}
    if TUNING_PATH.exists():
        with open(TUNING_PATH, encoding="utf-8") as f:
            tuning = json.load(f)

    for setting in settings:
        print(f"\n=== {setting}: {len(GRID)} (c1, c2) pairs, scored on fam_sh ===")
        start = time.time()
        tuning[setting] = tune_setting(setting, ru_vocab, be_vocab, dev)
        best = tuning[setting]["best"]
        print(f"Best: c1={best['c1']}, c2={best['c2']}, dev Macro-F1 {best['dev_macro_f1']:.4f} "
              f"({(time.time() - start) / 60:.1f} min)")

    with open(TUNING_PATH, "w", encoding="utf-8") as f:
        json.dump(tuning, f, indent=2)
    print(f"\nSaved {TUNING_PATH}")


if __name__ == "__main__":
    main()
