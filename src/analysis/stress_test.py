"""
Phase 3 stress test and Phase 4 summary tables, from the saved results/<run>.json files
(no new training).

1. Stress test: each model and setting from standard written language to speech
       Russian test          RU F1        (written Russian, in-domain for the treebank part)
       official Belarusian   BE F1        (written Belarusian, in-domain)
       Taraškievica          BE F1        (the other Belarusian spelling: orthographic shift)
       Trasyanka test        3-class and 4-class Macro-F1   (speech and code-mixing)
   The control sets are scored by the F1 of their own language rather than their Macro-F1:
   official Belarusian has 3 gold RU words, which pull its Macro-F1 to about 0.6 even when
   Belarusian itself is near perfect.
   Two drops summarise the table:
       spelling drop  Taraškievica BE F1 - official BE F1
       speech drop    Trasyanka 3-class Macro-F1 - mean(RU F1, official BE F1)
2. 3-class vs 4-class Macro-F1 on the Trasyanka test families. 3-class averages the F1 of
   BE, RU and SHARED only: HYBRID words stay in the data and a HYBRID prediction still counts
   as an error, but the HYBRID F1 itself is left out of the average. That is the fair way to
   compare zero-shot models, which cannot predict HYBRID at all.
3. Per-label precision / recall / F1 on the Trasyanka test families for each model's best run.

Prints the tables and writes results/stress_test.json.

Run from the project root:   python src/analysis/stress_test.py
"""
import re
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.baseline.splits import RESULTS_DIR, LEARNING_CURVE_FRACTIONS
from src.analysis.compare_models import run_name

# Table name -> run-name prefix (the tuned CRF stands for the CRF)
MODELS = {"CRF": "crf_tuned", "BiLSTM": "bilstm", "XLM-R": "xlmr", "SlavicBERT": "slavicbert"}
SETTINGS = [("zero_shot", 1.0), ("in_domain", 1.0), ("mixed", 1.0)]
ALL_RUNS = SETTINGS + [("mixed", fraction) for fraction in LEARNING_CURVE_FRACTIONS]
THREE_CLASS_LABELS = ("BE", "RU", "SHARED")
STRESS_PATH = RESULTS_DIR / "stress_test.json"

# One label line of a classification report:  "      SHARED     0.9395    0.9570    0.9482     28152"
REPORT_LINE = re.compile(r"^\s*(BE|RU|SHARED|HYBRID)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$")


def load_results(name):
    """results/<name>.json as a dict {set name: metrics}, or None if the run does not exist."""
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def report_scores(report):
    """
    Per-label scores from a saved classification report (the "report" string in results/*.json):
        {"BE": {"precision": 0.9221, "recall": 0.8978, "f1": 0.9098, "support": 10389}, ...}
    Only the label lines; the accuracy and average lines are skipped.
    """
    scores = {}
    for line in report.splitlines():
        match = REPORT_LINE.match(line)
        if match:
            label, precision, recall, f1, support = match.groups()
            scores[label] = {"precision": float(precision), "recall": float(recall),
                             "f1": float(f1), "support": int(support)}
    return scores


def three_class_macro_f1(scores):
    """
    Mean F1 of the THREE_CLASS_LABELS that have gold words (support > 0) in `scores`
    (from report_scores). None if none of them has.
        {"BE": f1 0.90, "RU": f1 0.92, "SHARED": f1 0.94, "HYBRID": f1 0.77}  ->  0.92
    """
    # "label in scores" first: and stops there, so a missing label is never looked up
    f1s = [scores[label]["f1"] for label in THREE_CLASS_LABELS
           if label in scores and scores[label]["support"] > 0]
    if not f1s:
        return None
    return sum(f1s) / len(f1s)


def stress_row(results):
    """
    One stress-test row for one run, from load_results(). Every value is a float, or None when
    the run has no such set (e.g. no "trasyanka" outside zero-shot) or the drop cannot be computed:
        {"ru_test":       RU F1 on russian_test,
         "be_official":   BE F1 on belarusian_official_test,
         "taraskievica":  BE F1 on taraskievica,
         "trasyanka_3":   three_class_macro_f1 on trasyanka_test,
         "trasyanka_4":   the saved macro_f1 on trasyanka_test,
         "spelling_drop": taraskievica - be_official,
         "speech_drop":   trasyanka_3 - mean(ru_test, be_official)}
    """
    def label_f1(set_name, label):
        """F1 of one label on one set, or None if the run has no such set or label."""
        if set_name not in results:
            return None
        scores = report_scores(results[set_name]["report"])
        return scores[label]["f1"] if label in scores else None

    has_trasyanka = "trasyanka_test" in results
    row = {"ru_test": label_f1("russian_test", "RU"),
           "be_official": label_f1("belarusian_official_test", "BE"),
           "taraskievica": label_f1("taraskievica", "BE"),
           "trasyanka_3": (three_class_macro_f1(report_scores(results["trasyanka_test"]["report"]))
                           if has_trasyanka else None),
           "trasyanka_4": results["trasyanka_test"]["macro_f1"] if has_trasyanka else None}
    # A drop needs every score it compares; one missing score makes it None
    row["spelling_drop"] = (row["taraskievica"] - row["be_official"]
                            if None not in (row["taraskievica"], row["be_official"]) else None)
    row["speech_drop"] = (row["trasyanka_3"] - (row["ru_test"] + row["be_official"]) / 2
                          if None not in (row["trasyanka_3"], row["ru_test"], row["be_official"]) else None)
    return row


def best_runs():
    """
    {table model name: (run name, its results)} for the run of each model in ALL_RUNS with the
    highest 4-class Macro-F1 on trasyanka_test. Runs whose file is missing are skipped.
    """
    best = {}       # model -> (run name, results) of the best run so far
    for model, prefix in MODELS.items():
        for setting, fraction in ALL_RUNS:
            name = run_name(prefix, setting, fraction)
            results = load_results(name)
            if results is None:
                continue
            score = results["trasyanka_test"]["macro_f1"]
            # > rather than >=: on a tie the earlier run in ALL_RUNS stays
            if model not in best or score > best[model][1]["trasyanka_test"]["macro_f1"]:
                best[model] = (name, results)
    return best


def fmt(value, signed=False):
    if value is None:
        return "   —  "
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def main():
    stress = {}
    print("Stress test: F1 of each set's own language, then Trasyanka Macro-F1 (3-class / 4-class)")
    header = (f"  {'run':<22}{'RU test':>9}{'BE off.':>9}{'Tarask.':>9}{'Tras. 3':>9}{'Tras. 4':>9}"
              f"{'spelling':>10}{'speech':>9}")
    for setting, fraction in SETTINGS:
        print(f"\n{setting}\n{header}")
        for model, prefix in MODELS.items():
            name = run_name(prefix, setting, fraction)
            results = load_results(name)
            if results is None:
                print(f"  {model:<22}  (no results)")
                continue
            row = stress_row(results)
            stress[name] = row
            print(f"  {model:<22}{fmt(row['ru_test']):>9}{fmt(row['be_official']):>9}{fmt(row['taraskievica']):>9}"
                  f"{fmt(row['trasyanka_3']):>9}{fmt(row['trasyanka_4']):>9}"
                  f"{fmt(row['spelling_drop'], signed=True):>10}{fmt(row['speech_drop'], signed=True):>9}")

    print("\n3-class vs 4-class Macro-F1 on the Trasyanka test families, every run")
    print(f"  {'run':<22}{'3-class':>9}{'4-class':>9}{'HYBRID F1':>11}")
    three_vs_four = {}
    for model, prefix in MODELS.items():
        for setting, fraction in ALL_RUNS:
            name = run_name(prefix, setting, fraction)
            results = load_results(name)
            if results is None:
                continue
            scores = report_scores(results["trasyanka_test"]["report"])
            row = {"three_class": three_class_macro_f1(scores),
                   "four_class": results["trasyanka_test"]["macro_f1"],
                   "hybrid_f1": scores["HYBRID"]["f1"]}
            three_vs_four[name] = row
            print(f"  {name:<22}{fmt(row['three_class']):>9}{fmt(row['four_class']):>9}{fmt(row['hybrid_f1']):>11}")

    print("\nPer-label scores on the Trasyanka test families, best run of each model")
    per_label = {}
    for model, (name, results) in best_runs().items():
        scores = report_scores(results["trasyanka_test"]["report"])
        per_label[name] = scores
        print(f"\n  {model} ({name}, Macro-F1 {results['trasyanka_test']['macro_f1']:.4f})")
        print(f"    {'label':<8}{'precision':>11}{'recall':>9}{'F1':>8}{'words':>9}")
        for label, s in scores.items():
            print(f"    {label:<8}{s['precision']:>11.3f}{s['recall']:>9.3f}{s['f1']:>8.3f}{s['support']:>9,}")

    with open(STRESS_PATH, "w", encoding="utf-8") as f:
        json.dump({"stress": stress, "three_vs_four": three_vs_four, "best_run_per_label": per_label}, f, indent=2)
    print(f"\nSaved {STRESS_PATH}")


if __name__ == "__main__":
    main()
