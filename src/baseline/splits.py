"""
Training and evaluation data shared by every model (baselines now, BiLSTM and transformers later).

Training (zero-shot setting): Russian train split + official-orthography Belarusian train
split, with Russian sampled down to the same number of sentences as Belarusian.

Evaluation sets:
    trasyanka                  Oldenburg corpus, gold labels (the main test)
    russian_test               SynTagRus test split (in-domain control)
    belarusian_official_test   HSE test split, official orthography (in-domain control)
    taraskievica               HSE Taraškievica sentences, never trained on (orthographic shift)
"""
import sys
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.build_processed import load_processed
from src.utils.metrics import evaluate_predictions, LABELS

RESULTS_DIR = PROJECT_ROOT / "results"
RANDOM_SEED = 0


def training_sentences(seed=RANDOM_SEED):
    """
    Zero-shot training set: official-orthography BE train plus an equally sized,
    seeded sample of RU train (Russian first). 16,704 records, 195,150 words.
    """
    be_records = load_processed("belarusian")
    be_train = [record for record in be_records if record["split"] == "train" and record["orthography"] == "official"]
    ru_records = load_processed("russian")
    ru_train = [record for record in ru_records if record["split"] == "train"]
    ru_train = random.Random(seed).sample(ru_train, len(be_train))
    return ru_train + be_train


def evaluation_sets():
    """
    The four evaluation sets as {name: records}: trasyanka (all), russian_test,
    belarusian_official_test, and taraskievica (tarask sentences from any split).
    """
    trasyanka = load_processed("trasyanka")
    be_records = load_processed("belarusian")
    be_official_test = [record for record in be_records if record["split"] == "test" and record["orthography"] == "official"]
    tarask = [record for record in be_records if record["orthography"] == "tarask"]
    ru_records = load_processed("russian")
    ru_test = [record for record in ru_records if record["split"] == "test"]
    return {"trasyanka": trasyanka,
            "russian_test": ru_test,
            "belarusian_official_test": be_official_test,
            "taraskievica": tarask}


def evaluate_on_sets(predict_sentences, sets):
    """
    Scores one model on every set. predict_sentences maps token lists to label lists.
    Macro-F1 averages only over the labels present in that set's gold data.
    Returns {name: metrics dict from metrics.py}.
    """
    results = {}
    for name, records in sets.items():
        gold = [record["labels"] for record in records]
        pred = predict_sentences([record["tokens"] for record in records])
        # A label with no gold examples (e.g. HYBRID outside Trasyanka) would score 0 and drag the average
        labels = [label for label in LABELS if any(label in sentence_labels for sentence_labels in gold)]
        results[name] = evaluate_predictions(gold, pred, labels=labels)
    return results


def print_macro_f1_table(model_name, results):
    """Prints one line per evaluation set: Macro-F1, and the labels it averages over."""
    print(f"\n{model_name}")
    for name, metrics in results.items():
        print(f"  {name:<26} Macro-F1 {metrics['macro_f1']:.4f}   ({', '.join(metrics['labels'])})")


def save_results(model_name, results):
    """Writes results/<model_name>.json (Macro-F1 and per-label report per set) for later comparison."""
    RESULTS_DIR.mkdir(exist_ok=True)
    summary = {
        name: {
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "labels": metrics["labels"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
            "report": metrics["report"],
        }
        for name, metrics in results.items()
    }
    path = RESULTS_DIR / f"{model_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved {path}")
