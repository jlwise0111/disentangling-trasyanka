"""
Training and evaluation data shared by every model (baselines now, BiLSTM and transformers later).

Training settings (training_sentences):
    zero_shot   Russian train split + official-orthography Belarusian train split, with
                Russian sampled down to the same number of sentences as Belarusian
    in_domain   Oldenburg train families only
    mixed       both
The Oldenburg corpus is split by family into train / dev / test (constants below);
the learning curve subsamples the train families by transcript.

Evaluation sets:
    trasyanka                  Oldenburg corpus, all families (zero-shot only)
    trasyanka_test             Oldenburg test families (the main test for every setting)
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
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
RANDOM_SEED = 0

# ---------------------------------------------------------------------------
# Oldenburg corpus split, by family (never by utterance: the same speakers and
# conversations would end up on both sides).
#
# family    utterances   words   HYBRID
# fam_ak       4,891    23,415     684
# fam_ba       5,192    24,490     433
# fam_ch       8,406    33,144     622
# fam_mi       5,446    32,398     407
# fam_ra       5,509    31,985     772
# fam_sh       5,406    23,069     396
# fam_sm       5,848    32,256     831
#
# Test: fam_ba + fam_ra = 56,475 words (28%), HYBRID 2.1% (whole corpus 2.1%).
# Dev:  fam_sh, the smallest training family, used only for early stopping / tuning.
# Train: the other four = 121,213 words, 2,544 HYBRID.
# ---------------------------------------------------------------------------
TRASYANKA_FAMILIES = ["fam_ak", "fam_ba", "fam_ch", "fam_mi", "fam_ra", "fam_sh", "fam_sm"]
TEST_FAMILIES = ["fam_ba", "fam_ra"]
DEV_FAMILY = "fam_sh"
TRAIN_FAMILIES = [f for f in TRASYANKA_FAMILIES if f not in TEST_FAMILIES and f != DEV_FAMILY]

TRAINING_SETTINGS = ("zero_shot", "in_domain", "mixed")
LEARNING_CURVE_FRACTIONS = (0.1, 0.3, 0.7)


def trasyanka_split(part):
    """
    Trasyanka records for one part of the family split: "train", "dev" or "test".
    Raises ValueError for any other part.
    """
    data = load_processed("trasyanka")

    if part == "train":
        return [record for record in data if record["family"] in TRAIN_FAMILIES]
    elif part == "dev":
        return [record for record in data if record["family"] in [DEV_FAMILY]]
    elif part == "test":
        return [record for record in data if record["family"] in TEST_FAMILIES]
    else:
        raise ValueError("Invalid part.")


def sample_by_transcript(records, fraction, seed=RANDOM_SEED):
    """
    Keeps about `fraction` of the records by sampling whole transcripts (the "source"
    field, e.g. fam_ak_1.cha), so a conversation is never cut in half.
    fraction=1.0 returns all records. Same seed -> same transcripts.
    """
    # Sorted, because set order is not stable between runs and the seed must give the same sample
    sources = sorted({record["source"] for record in records})
    n = max(1, round(len(sources) * fraction))
    sample = set(random.Random(seed).sample(sources, n))
    return [record for record in records if record["source"] in sample]


def treebank_training_sentences(seed=RANDOM_SEED):
    """
    Official-orthography BE train plus an equally sized, seeded sample of RU train
    (Russian first). 16,704 records, 195,150 words.
    """
    be_records = load_processed("belarusian")
    be_train = [record for record in be_records if record["split"] == "train" and record["orthography"] == "official"]
    ru_records = load_processed("russian")
    ru_train = [record for record in ru_records if record["split"] == "train"]
    ru_train = random.Random(seed).sample(ru_train, len(be_train))
    return ru_train + be_train


def training_sentences(setting="zero_shot", fraction=1.0, seed=RANDOM_SEED):
    """
    Training records for one setting:
        zero_shot   treebanks only (fraction is ignored)
        in_domain   Trasyanka train families only
        mixed       treebanks + Trasyanka train families
    fraction < 1 subsamples the Trasyanka part by transcript (the learning curve).
    Defaults reproduce the old zero-shot training set, so existing callers are unchanged.
    """
    if setting == "zero_shot":
        return treebank_training_sentences(seed=seed)
    elif setting == "in_domain":
        return sample_by_transcript(trasyanka_split("train"), fraction, seed)
    elif setting == "mixed":
        return treebank_training_sentences(seed=seed) + sample_by_transcript(trasyanka_split("train"), fraction, seed)
    else:
        raise ValueError("Invalid setting")


def evaluation_sets():
    """
    The evaluation sets as {name: records}: trasyanka (all families, kept for the
    zero-shot Checkpoint 1 numbers), trasyanka_test (test families only, the set every
    setting is compared on), russian_test, belarusian_official_test, and taraskievica
    (tarask sentences from any split).
    """
    trasyanka = load_processed("trasyanka")
    be_records = load_processed("belarusian")
    be_official_test = [record for record in be_records if record["split"] == "test" and record["orthography"] == "official"]
    tarask = [record for record in be_records if record["orthography"] == "tarask"]
    ru_records = load_processed("russian")
    ru_test = [record for record in ru_records if record["split"] == "test"]
    return {"trasyanka": trasyanka,
            "trasyanka_test": trasyanka_split("test"),
            "russian_test": ru_test,
            "belarusian_official_test": be_official_test,
            "taraskievica": tarask}


def evaluate_on_sets(predict_sentences, sets, field="tokens"):
    """
    Scores one model on every set. predict_sentences maps word lists to label lists;
    the words come from each record's `field`: "tokens" (Latin, the default) or "cyrillic".
    Macro-F1 averages only over the labels present in that set's gold data.
    Returns {name: metrics dict from metrics.py}; each metrics dict also keeps the
    predicted label lists under "predictions" (for save_predictions; save_results skips them).
    """
    results = {}
    for name, records in sets.items():
        gold = [record["labels"] for record in records]
        pred = predict_sentences([record[field] for record in records])
        # A label with no gold examples (e.g. HYBRID outside Trasyanka) would score 0 and drag the average
        labels = [label for label in LABELS if any(label in sentence_labels for sentence_labels in gold)]
        results[name] = evaluate_predictions(gold, pred, labels=labels)
        results[name]["predictions"] = [list(sentence) for sentence in pred]
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


def save_predictions(model_name, sets, results, set_name="trasyanka_test"):
    """
    Writes results/predictions/<model_name>.jsonl: one line per sentence of `set_name`
    with its id, family, tokens, gold labels and predicted labels, for word-level analysis
    (in-vocabulary vs out-of-vocabulary words, error audit).
    `sets` and `results` are what evaluate_on_sets was given and returned.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    records = sets[set_name]
    predictions = results[set_name]["predictions"]
    assert len(records) == len(predictions), "predictions do not match the set's sentences"
    path = PREDICTIONS_DIR / f"{model_name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for record, predicted in zip(records, predictions):
            line = {"id": record["id"], "family": record["family"], "tokens": record["tokens"],
                    "labels": record["labels"], "predicted": predicted}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"Saved {path}")
