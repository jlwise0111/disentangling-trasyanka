"""
Checkpoint 2: the CRF and the BiLSTM on the Trasyanka test families, split into
in-vocabulary words (the word form occurs in that run's training data) and
out-of-vocabulary words (it does not). Character modelling should matter most on
out-of-vocabulary words, e.g. spoken Russian written as pronounced (nada, tožy).

Reads results/predictions/<run>.jsonl (written by train_crf.py and train_bilstm.py),
prints one table per setting and writes results/comparison.json.
Runs without a prediction file yet are skipped.

Run from the project root:   python src/analysis/compare_models.py
"""
import sys
import json
from pathlib import Path

from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.metrics import LABELS
from src.baseline.splits import training_sentences, RESULTS_DIR, PREDICTIONS_DIR, LEARNING_CURVE_FRACTIONS

MODELS = ["crf", "bilstm"]
RUNS = ([("zero_shot", 1.0), ("in_domain", 1.0), ("mixed", 1.0)]
        + [("mixed", fraction) for fraction in LEARNING_CURVE_FRACTIONS])
COMPARISON_PATH = RESULTS_DIR / "comparison.json"


def run_name(model, setting, fraction):
    """crf, crf_in_domain, bilstm_mixed_30 ... (the names the training scripts use)"""
    if setting == "zero_shot":
        return model
    name = f"{model}_{setting}"
    if fraction != 1.0:
        name += f"_{round(fraction * 100)}"
    return name


def load_predictions(name):
    """The records of results/predictions/<name>.jsonl, or None if that run has not been made yet."""
    path = PREDICTIONS_DIR / f"{name}.jsonl"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def training_vocabulary(setting, fraction):
    """
    The set of word forms (the Latin "tokens") in the training data of this setting.
    Both models in a setting train on exactly the same sentences, so they share it.
    """
    records = training_sentences(setting, fraction)
    # A set, not a list: duplicates collapse and "token in vocabulary" is a fast lookup
    return {token for record in records for token in record["tokens"]}


def split_by_vocabulary(records, vocabulary):
    """
    Splits every word of the prediction records into two groups by whether its token is
    in the vocabulary. Returns two dicts, (in_vocab, out_of_vocab), each
        {"gold": [label, label, ...], "predicted": [label, label, ...]}
    as flat lists over all sentences, word by word in the same order.
    """
    in_vocab = {"gold": [], "predicted": []}
    out_of_vocab = {"gold": [], "predicted": []}
    for record in records:
        for token, gold, predicted in zip(record["tokens"], record["labels"], record["predicted"]):
            group = in_vocab if token in vocabulary else out_of_vocab
            group["gold"].append(gold)
            group["predicted"].append(predicted)

    return in_vocab, out_of_vocab


def score(group):
    """
    Macro-F1 over the labels present in the group's gold labels, HYBRID F1 (None if the group
    has no gold HYBRID word) and the number of words.
    """
    gold, predicted = group["gold"], group["predicted"]
    if not gold:
        return {"words": 0, "macro_f1": None, "hybrid_f1": None}
    labels = [label for label in LABELS if label in set(gold)]
    macro = f1_score(gold, predicted, labels=labels, average="macro", zero_division=0)
    hybrid = (f1_score(gold, predicted, labels=["HYBRID"], average="macro", zero_division=0)
              if "HYBRID" in labels else None)
    return {"words": len(gold), "macro_f1": round(macro, 4),
            "hybrid_f1": None if hybrid is None else round(hybrid, 4)}


def fmt(value):
    return "  —   " if value is None else f"{value:.4f}"


def main():
    """Scores every available run on all / in-vocabulary / out-of-vocabulary words and saves the table."""
    comparison = {}
    for setting, fraction in RUNS:
        runs = {model: load_predictions(run_name(model, setting, fraction)) for model in MODELS}
        if all(records is None for records in runs.values()):
            continue
        vocabulary = training_vocabulary(setting, fraction)
        label = setting if fraction == 1.0 else f"{setting} {round(fraction * 100)}%"
        print(f"\n{label}  ({len(vocabulary):,} training word forms)")
        print(f"  {'model':<8}{'words':>8}{'all F1':>9}{'IV F1':>9}{'OOV F1':>9}{'OOV share':>11}"
              f"{'HYB IV':>9}{'HYB OOV':>9}")
        for model, records in runs.items():
            if records is None:
                print(f"  {model:<8}  (no predictions yet)")
                continue
            in_vocab, out_of_vocab = split_by_vocabulary(records, vocabulary)
            everything = {"gold": in_vocab["gold"] + out_of_vocab["gold"],
                          "predicted": in_vocab["predicted"] + out_of_vocab["predicted"]}
            row = {"all": score(everything), "in_vocab": score(in_vocab), "out_of_vocab": score(out_of_vocab)}
            comparison[run_name(model, setting, fraction)] = row
            oov_share = row["out_of_vocab"]["words"] / row["all"]["words"]
            print(f"  {model:<8}{row['all']['words']:>8,}{fmt(row['all']['macro_f1']):>9}"
                  f"{fmt(row['in_vocab']['macro_f1']):>9}{fmt(row['out_of_vocab']['macro_f1']):>9}"
                  f"{oov_share:>10.1%} {fmt(row['in_vocab']['hybrid_f1']):>9}{fmt(row['out_of_vocab']['hybrid_f1']):>9}")

    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
