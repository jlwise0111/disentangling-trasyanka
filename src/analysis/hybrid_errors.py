"""
Phase 4: where the models fail on HYBRID words (a Russian stem with a Belarusian ending or the
other way round, e.g. atrézaú = otrez- + -aú).

Uses the mixed runs of all four models (the setting where one model has to handle speech and
both standard languages) on the Trasyanka test families:
    1. a HYBRID confusion per model: what missed hybrid words were called instead, and which
       gold labels were wrongly called HYBRID;
    2. for every gold HYBRID word, which models missed it; words every model misses are the
       hard cases rather than one model's weakness;
    3. the hybrid word forms missed most often, ranked, written to an audit sheet with their
       sentence, the %mor BE / RU lemmas, POS and features, how the form is labeled in the
       training data, and empty columns for the native-speaker partner (AUDIT_COLUMNS).

Training status of a form, from the mixed training data the four models saw (TRAINING):
    unseen            the form never occurs in training
    always HYBRID     every training occurrence is HYBRID
    sometimes HYBRID  HYBRID and other labels (the same spelling, different words or uses)
    never HYBRID      it occurs, but never as HYBRID

No confidence scores yet: the ranking uses how many models miss a word and how often.

Writes results/hybrid_errors.json (counts) and results/hybrid_audit.csv (the audit sheet,
UTF-8 with BOM so Excel shows Cyrillic and diacritics correctly).

Run from the project root:   python src/analysis/hybrid_errors.py
"""
import sys
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.baseline.splits import trasyanka_split, training_sentences, RESULTS_DIR
from src.analysis.compare_models import load_predictions

# Short model name -> its mixed run (100 % of the Trasyanka training data)
AUDIT_RUNS = {"crf": "crf_tuned_mixed", "bilstm": "bilstm_mixed",
              "xlmr": "xlmr_mixed", "slavicbert": "slavicbert_mixed"}
TRAINING = ("mixed", 1.0)       # the (setting, fraction) the audited runs were trained on
TRAIN_STATUSES = ("unseen", "always HYBRID", "sometimes HYBRID", "never HYBRID")
TOP_N = 20
# Filled in during the audit, one answer per row; fixed answers so they can be counted afterwards
AUDIT_COLUMNS = {
    "is_hybrid": "yes / no / unsure: is this word HYBRID in this sentence?",
    "stem_language": "RU / BE / unclear: the language of the stem",
    "mixed_part": "ending / sounds / both / other: what comes from the other language",
    "notes": "free text",
}
SUMMARY_PATH = RESULTS_DIR / "hybrid_errors.json"
AUDIT_PATH = RESULTS_DIR / "hybrid_audit.csv"


def load_runs(test_records):
    """
    {model: prediction records} for AUDIT_RUNS. Checks that every run lists the test sentences
    in the same order as trasyanka_split("test"), so sentence i and word j mean the same word in
    every run. Raises FileNotFoundError for a missing run.
    """
    runs = {}
    for model, name in AUDIT_RUNS.items():
        records = load_predictions(name)
        if records is None:
            raise FileNotFoundError(f"results/predictions/{name}.jsonl is missing")
        assert [r["id"] for r in records] == [r["id"] for r in test_records], f"{name}: sentence order differs"
        runs[model] = records
    return runs


def training_label_counts(setting=TRAINING[0], fraction=TRAINING[1]):
    """{token: Counter of its gold labels} over the training data of one setting."""
    counts = defaultdict(Counter)
    for record in training_sentences(setting, fraction):
        for token, label in zip(record["tokens"], record["labels"]):
            counts[token][label] += 1
    return counts


def train_status(token, label_counts):
    """One of TRAIN_STATUSES for a token, from training_label_counts()."""
    labels = label_counts.get(token)        # .get: looking up a missing key must not add it
    if not labels:
        return "unseen"
    if "HYBRID" not in labels:
        return "never HYBRID"
    return "always HYBRID" if len(labels) == 1 else "sometimes HYBRID"


def format_counts(counter):
    """'SHARED 6, HYBRID 1' (most common first), '' for an empty counter."""
    return ", ".join(f"{label} {n}" for label, n in counter.most_common())


def hybrid_confusion(records):
    """
    HYBRID errors of one model, from its prediction records (each with "labels" and "predicted").
    Returns
        {"gold_hybrid": number of gold HYBRID words,
         "correct": how many of them were predicted HYBRID,
         "missed_as": Counter {predicted label: count} for gold HYBRID words predicted otherwise,
         "false_hybrid": Counter {gold label: count} for words predicted HYBRID whose gold is not HYBRID}
    e.g. one sentence, labels [HYBRID, HYBRID, BE], predicted [HYBRID, RU, HYBRID]
        -> gold_hybrid 2, correct 1, missed_as {RU: 1}, false_hybrid {BE: 1}
    """
    counts = {"gold_hybrid": 0, "correct": 0, "missed_as": Counter(), "false_hybrid": Counter()}
    for record in records:
        for gold, predicted in zip(record["labels"], record["predicted"]):
            if gold == "HYBRID":
                counts["gold_hybrid"] += 1
                if predicted == "HYBRID":
                    counts["correct"] += 1
                else:
                    counts["missed_as"][predicted] += 1     # what the model called it instead
            elif predicted == "HYBRID":                     # elif: a correct HYBRID is not a false one
                counts["false_hybrid"][gold] += 1           # what the word really was
    return counts


def hybrid_word_errors(test_records, runs):
    """
    One entry per gold HYBRID word in the test set, in test order:
        {"sentence": i, "word": j, "token": the Latin token,
         "predictions": {model: predicted label, ...}  (every model in runs, same order),
         "missed_by": [models whose prediction is not HYBRID]}
    Correctly labeled words are included too (missed_by == []), so the counts by number of
    models are complete.
    """
    errors = []
    for i, record in enumerate(test_records):
        for j, gold in enumerate(record["labels"]):
            if gold != "HYBRID":
                continue
            # load_runs checked the sentence order, so sentence i, word j is the same word in every run
            predictions = {model: records[i]["predicted"][j] for model, records in runs.items()}
            missed_by = [model for model, predicted in predictions.items() if predicted != "HYBRID"]
            errors.append({"sentence": i, "word": j, "token": record["tokens"][j],
                           "predictions": predictions, "missed_by": missed_by})
    return errors


def rank_word_forms(errors, top_n=TOP_N):
    """
    The hybrid word forms (tokens) that are missed most, as a list of up to top_n dicts:
        {"token": ..., "occurrences": how many times it is gold HYBRID in the test set,
         "missed_by_all": how many of those occurrences every model missed,
         "first_missed_by_all": the error entry of its first occurrence that every model
                                missed (None if there is none)}
    Only forms missed by every model at least once are ranked. Sorted by missed_by_all
    (most first), then occurrences (most first), then token (alphabetically), so ties always
    come out in the same order.
    """
    forms = {}      # token -> its entry, in order of first occurrence
    for error in errors:
        token = error["token"]
        if token not in forms:
            forms[token] = {"token": token, "occurrences": 0, "missed_by_all": 0, "first_missed_by_all": None}
        form = forms[token]
        form["occurrences"] += 1
        if len(error["missed_by"]) == len(error["predictions"]):      # every model missed it
            form["missed_by_all"] += 1
            if form["first_missed_by_all"] is None:                   # errors are in test order: keep the first
                form["first_missed_by_all"] = error
    ranked = [form for form in forms.values() if form["missed_by_all"] > 0]
    # Minus signs: sort puts small first, so -count puts the most-missed forms first
    ranked.sort(key=lambda form: (-form["missed_by_all"], -form["occurrences"], form["token"]))
    return ranked[:top_n]


def sentence_with_mark(words, j):
    """The sentence as one string with word j in brackets: 'ja [prykalvaju] vaabšče'."""
    return " ".join(f"[{word}]" if k == j else word for k, word in enumerate(words))


def audit_rows(ranked, test_records, label_counts):
    """
    One row per ranked word form, with the context the audit needs and empty columns to fill in.
    label_counts comes from training_label_counts().
    """
    rows = []
    for rank, form in enumerate(ranked, start=1):
        error = form["first_missed_by_all"]
        record = test_records[error["sentence"]]
        j = error["word"]
        token = form["token"]
        rows.append({
            "rank": rank,
            "token": token,
            "cyrillic": record["cyrillic"][j],
            "raw": record["raw"][j],                       # with the reduced-vowel marks
            "occurrences": form["occurrences"],
            "missed_by_all": form["missed_by_all"],
            "train_status": train_status(token, label_counts),
            "train_labels": format_counts(label_counts.get(token, Counter())),
            **{f"pred_{model}": label for model, label in error["predictions"].items()},
            "be_lemma": record["be_lemma"][j] or "",
            "ru_lemma": record["ru_lemma"][j] or "",
            "pos": record["pos"][j] or "",
            "feats": record["feats"][j] or "",
            "sentence_id": record["id"],
            "sentence": sentence_with_mark(record["tokens"], j),
            **{column: "" for column in AUDIT_COLUMNS},   # filled in during the audit
        })
    return rows


def write_audit_sheet(rows, path=AUDIT_PATH):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def main():
    test_records = trasyanka_split("test")
    runs = load_runs(test_records)

    print("HYBRID on the test families, mixed setting")
    print(f"  {'model':<11}{'gold':>6}{'correct':>9}{'recall':>8}   missed as{'':<22}false HYBRID")
    confusions = {}
    for model, records in runs.items():
        c = hybrid_confusion(records)
        confusions[model] = c
        missed, false = format_counts(c["missed_as"]), format_counts(c["false_hybrid"])
        print(f"  {model:<11}{c['gold_hybrid']:>6}{c['correct']:>9}{c['correct'] / c['gold_hybrid']:>8.2f}"
              f"   {missed:<31}{false} ({sum(c['false_hybrid'].values())})")

    errors = hybrid_word_errors(test_records, runs)
    label_counts = training_label_counts()
    by_count = Counter(len(error["missed_by"]) for error in errors)
    # {number of models missing the word: Counter of the training status of its form}
    status_by_count = defaultdict(Counter)
    for error in errors:
        status_by_count[len(error["missed_by"])][train_status(error["token"], label_counts)] += 1
    print(f"\nGold HYBRID words by how many of the {len(runs)} models miss them, and the training status of the form")
    print(f"  {'':<12}{'words':>7}{'':<7}" + "".join(f"{status:>18}" for status in TRAIN_STATUSES))
    for k in range(len(runs) + 1):
        shares = "".join(f"{status_by_count[k][status] / max(by_count[k], 1):>18.0%}" for status in TRAIN_STATUSES)
        print(f"  missed by {k}:{by_count[k]:>7} ({by_count[k] / len(errors):>4.0%}){shares}")

    ranked = rank_word_forms(errors)
    print(f"\nTop {len(ranked)} hybrid forms missed by every model")
    print(f"  {'token':<18}{'occurrences':>12}{'missed by all':>15}   {'training':<18}predictions (first such occurrence)")
    for form in ranked:
        predictions = " ".join(f"{model}={label}" for model, label in
                               form["first_missed_by_all"]["predictions"].items())
        print(f"  {form['token']:<18}{form['occurrences']:>12}{form['missed_by_all']:>15}   "
              f"{train_status(form['token'], label_counts):<18}{predictions}")

    summary = {
        "runs": AUDIT_RUNS,
        "training": {"setting": TRAINING[0], "fraction": TRAINING[1]},
        "confusion": {model: {"gold_hybrid": c["gold_hybrid"], "correct": c["correct"],
                              "missed_as": dict(c["missed_as"]), "false_hybrid": dict(c["false_hybrid"])}
                      for model, c in confusions.items()},
        "missed_by_count": {str(k): by_count[k] for k in range(len(runs) + 1)},
        "train_status_by_missed_count": {str(k): dict(status_by_count[k]) for k in range(len(runs) + 1)},
        "top_forms": [{"token": form["token"], "occurrences": form["occurrences"],
                       "missed_by_all": form["missed_by_all"], "train_status": train_status(form["token"], label_counts)}
                      for form in ranked],
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {SUMMARY_PATH}")
    write_audit_sheet(audit_rows(ranked, test_records, label_counts))


if __name__ == "__main__":
    main()
