"""
CRF baseline: a linear-chain CRF (sklearn-crfsuite) trained in one of the training
settings from splits.py (zero_shot, in_domain, mixed) and evaluated on every evaluation set.

Run from the project root:
    python src/baseline/train_crf.py                                   # zero-shot, results/crf.json
    python src/baseline/train_crf.py --setting in_domain               # results/crf_in_domain.json
    python src/baseline/train_crf.py --setting mixed --fraction 0.3    # results/crf_mixed_30.json
    python src/baseline/train_crf.py --all                             # every run in ALL_RUNS
    python src/baseline/train_crf.py --all --tuned                     # c1 / c2 from tune_crf.py, results/crf_tuned*.json
Zero-shot takes about 2 minutes (training itself ~45 s); mixed takes longer, it has more training data.
"""
import sys
import json
import time
import argparse
from pathlib import Path

import sklearn_crfsuite

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.conllu_parser import load_both_vocabs
from src.baseline.crf_features import sent_to_features
from src.baseline.splits import (training_sentences, evaluation_sets, evaluate_on_sets,
                                 print_macro_f1_table, save_results, save_predictions,
                                 TRAINING_SETTINGS, LEARNING_CURVE_FRACTIONS, RESULTS_DIR)

CRF_PARAMS = {
    "algorithm": "lbfgs",
    "c1": 0.1,                       # L1 penalty
    "c2": 0.1,                       # L2 penalty
    "max_iterations": 100,
    "all_possible_transitions": True,
}
TUNING_PATH = RESULTS_DIR / "crf_tuning.json"      # written by tune_crf.py

# (setting, fraction) pairs run by --all: the three settings on all training families,
# then the learning curve (treebanks + part of the Trasyanka training families).
ALL_RUNS = ([("zero_shot", 1.0), ("in_domain", 1.0), ("mixed", 1.0)]
            + [("mixed", fraction) for fraction in LEARNING_CURVE_FRACTIONS])


def run_name(setting, fraction, tuned=False):
    """
    Name used for printing and for results/<name>.json:
        zero_shot, any fraction  -> "crf"            (keeps the Checkpoint 1 file name)
        in_domain, 1.0           -> "crf_in_domain"
        mixed, 1.0               -> "crf_mixed"
        mixed, 0.3               -> "crf_mixed_30"   (percent, as a whole number)
    With tuned=True the prefix is "crf_tuned" instead of "crf" (crf_tuned, crf_tuned_mixed_30 ...).
    """
    prefix = "crf_tuned" if tuned else "crf"
    if setting == "zero_shot":
        return prefix
    name = f"{prefix}_{setting}"
    if fraction != 1.0:
        name += f"_{round(fraction * 100)}"
    return name


def tuned_params(setting):
    """
    CRF_PARAMS with the best c1 / c2 that tune_crf.py found for this setting on fam_sh.
    Tuning ran at fraction 1.0, so the learning-curve runs use the mixed setting's pair.
    """
    if not TUNING_PATH.exists():
        raise FileNotFoundError(f"{TUNING_PATH} not found: run src/baseline/tune_crf.py first")
    with open(TUNING_PATH, encoding="utf-8") as f:
        best = json.load(f)[setting]["best"]
    return {**CRF_PARAMS, "c1": best["c1"], "c2": best["c2"]}


def train_and_evaluate(setting, fraction, ru_vocab, be_vocab, sets, params=CRF_PARAMS):
    """
    Trains one CRF with `params` on training_sentences(setting, fraction) and scores it on `sets`.
    Returns {set name: metrics} as evaluate_on_sets does.

    For in_domain and mixed, the full "trasyanka" set contains the training families,
    so it must be left out of the evaluation: only trasyanka_test is a fair Trasyanka score.
    """
    training_data = training_sentences(setting, fraction)
    X_train = [sent_to_features(record["tokens"], ru_vocab, be_vocab) for record in training_data]
    y_train = [record["labels"] for record in training_data]

    crf = sklearn_crfsuite.CRF(**params)
    start = time.time()
    crf.fit(X_train, y_train)
    print(f"Trained on {len(training_data):,} sentences in {time.time() - start:.0f}s")

    # Defined here, so it can use crf and the vocabularies; evaluate_on_sets calls it once per set
    def predict_sentences(token_lists):
        """Token lists -> predicted label lists."""
        return crf.predict([sent_to_features(tokens, ru_vocab, be_vocab) for tokens in token_lists])

    if setting != "zero_shot":
        # A new dict without the full Trasyanka set (it contains the training families)
        sets = {name: records for name, records in sets.items() if name != "trasyanka"}

    return evaluate_on_sets(predict_sentences, sets)


def main():
    """Parses the arguments, runs the CRF for each requested (setting, fraction) and saves results/<run_name>.json."""
    parser = argparse.ArgumentParser(description="Train and evaluate the CRF baseline.")
    parser.add_argument("--setting", choices=TRAINING_SETTINGS, default="zero_shot")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="share of the Trasyanka training transcripts (in_domain / mixed only)")
    parser.add_argument("--all", action="store_true", help="run every (setting, fraction) in ALL_RUNS")
    parser.add_argument("--tuned", action="store_true",
                        help="use the best c1 / c2 per setting from results/crf_tuning.json")
    args = parser.parse_args()

    runs = ALL_RUNS if args.all else [(args.setting, args.fraction)]

    # Loaded once and shared by every run
    ru_vocab, be_vocab = load_both_vocabs()
    sets = evaluation_sets()

    for setting, fraction in runs:
        name = run_name(setting, fraction, args.tuned)
        params = tuned_params(setting) if args.tuned else CRF_PARAMS
        print(f"\n=== {name}: setting={setting}, fraction={fraction}, c1={params['c1']}, c2={params['c2']} ===")
        results = train_and_evaluate(setting, fraction, ru_vocab, be_vocab, sets, params)
        print_macro_f1_table(name, results)
        print(results["trasyanka_test"]["report"])
        save_results(name, results)
        save_predictions(name, sets, results)


if __name__ == "__main__":
    main()
