"""
Builds the processed datasets from data/raw:

    data/processed/russian.jsonl      RU UD treebank (SynTagRus), pseudo-labels
    data/processed/belarusian.jsonl   BE UD treebank (HSE), pseudo-labels
    data/processed/trasyanka.jsonl    Oldenburg .cha corpus, gold labels
    data/processed/stats.json         sentence / token / label counts per dataset

One JSON object per line, one line per sentence. Word-level fields are stored as
parallel lists ("columns"), so sentence["tokens"][i] and sentence["labels"][i]
belong to the same word. "tokens" is always the shared Latin form (the model input).

Run from the project root:
    python src/utils/build_processed.py                       # all three datasets
    python src/utils/build_processed.py --only trasyanka      # just some of them
    python src/utils/build_processed.py --keep-unlabeled      # keep @u/@k/persname words (label null)
"""

import json
import re
import sys
import argparse
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.conllu_parser import process_treebank, load_both_vocabs, RU_FILES, BE_FILES
from src.utils.cha_parser import load_all_cha_files

OUT_DIR = "data/processed"
DATASETS = ["russian", "belarusian", "trasyanka"]

# Dataset name -> (language, conllu files)
TREEBANKS = {"russian": ("RU", RU_FILES), "belarusian": ("BE", BE_FILES)}

# One Cyrillic letter, including the Belarusian/Russian letters outside the а-я range
CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁіІўЎґҐ]")

# Output column name -> key in the parsers' token dicts.
# "tokens" is always the shared Latin form, the model input.
TREEBANK_FIELDS = {
    "tokens": "latin",
    "labels": "lang_label",
    "cyrillic": "form",         # original spelling, kept for reference
    "lemma": "lemma",
    "upos": "upos",
    "feats": "feats",
}

TRASYANKA_FIELDS = {
    "tokens": "latin",
    "labels": "lang_label",
    "raw": "raw",               # transcription with its reduced-vowel marks
    "phon_labels": "phon_label",
    "flags": "flags",
    "pos": "pos",
    "be_lemma": "be_lemma",
    "ru_lemma": "ru_lemma",
    "feats": "feats",
}


def is_kept_token(tok):
    """
    True if a treebank token belongs in the export. Drops what never occurs in
    spoken Trasyanka: hashtags, Latin-script words and numbers written as digits.
    """
    return CYRILLIC_RE.search(tok["form"]) is not None and not tok["form"].startswith("#")


def columns(tokens, fields):
    """
    Turns a list of token dicts into parallel lists, one per output field:
    [{"latin": "u"}, {"latin": "mne"}] with {"tokens": "latin"} -> {"tokens": ["u", "mne"]}.
    """
    return {out_name: [token[token_key] for token in tokens] for out_name, token_key in fields.items()}


def build_treebank(name, ru_vocab, be_vocab):
    """
    Builds one record per sentence of a treebank ("russian" or "belarusian"):
    sentence metadata plus TREEBANK_FIELDS columns. Sentences left empty by
    is_kept_token are skipped.
    """
    lang, files = TREEBANKS[name]
    records = []
    for file in files:
        sentence_dicts = process_treebank(file, lang, ru_vocab, be_vocab, with_metadata=True)
        for sentence in sentence_dicts:
            kept = [token for token in sentence["tokens"] if is_kept_token(token)]
            if not kept:
                continue
            records.append({
                "id": sentence["id"],
                "source": sentence["source"],
                "split": sentence["split"],
                "lang": sentence["lang"],
                "orthography": sentence["orthography"],
                "text": sentence["text"],
                **columns(kept, TREEBANK_FIELDS),
            })

    return records


def build_trasyanka(keep_unlabeled=False):
    """
    Builds one record per Trasyanka utterance: utterance metadata plus
    TRASYANKA_FIELDS columns. keep_unlabeled is passed on to cha_parser.
    """
    utterances = load_all_cha_files("data/raw", keep_unlabeled=keep_unlabeled)
    records = []
    for utterance in utterances:
        records.append({
            "id": utterance["id"],
            "source": utterance["source"],
            "family": utterance["family"],
            "speaker": utterance["speaker"],
            "mor_aligned": utterance["mor_aligned"],
            **columns(utterance["tokens"], TRASYANKA_FIELDS),
        })
    return records


def write_jsonl(records, path):
    """
    Writes records to a .jsonl file (path relative to PROJECT_ROOT), one JSON object per line.
    ensure_ascii=False keeps "ničoha" readable instead of escaping it.
    """
    path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records):
    """Counts sentences, words and labels (most common first) for stats.json."""
    return {
        "sentences": len(records),
        "tokens":    sum(len(record["tokens"]) for record in records),
        "labels":    dict(Counter(label for record in records for label in record["labels"]).most_common()),
    }


def load_processed(name):
    """
    Reads data/processed/<name>.jsonl back as a list of dicts.
    The baselines and models load their data through this instead of re-parsing data/raw.
    """
    path = PROJECT_ROOT / OUT_DIR / f"{name}.jsonl"

    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_args():
    """Command-line options: which datasets to build, and whether to keep unlabeled Trasyanka words."""
    parser = argparse.ArgumentParser(description="Build data/processed/*.jsonl and stats.json from data/raw.")
    parser.add_argument("--only", nargs="+", choices=DATASETS, default=DATASETS,
                        help="build just these datasets (default: all three)")
    parser.add_argument("--keep-unlabeled", action="store_true",
                        help="keep Trasyanka words without one of the four labels (label null)")
    return parser.parse_args()


def main():
    """Builds the requested datasets, writes their .jsonl files, and updates stats.json."""
    args = parse_args()
    stats_path = PROJECT_ROOT / OUT_DIR / "stats.json"

    # With --only, keep the counts of the datasets that are not rebuilt this time
    stats = {}
    if args.only != DATASETS and stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

    # Vocabularies are only needed for the treebanks' pseudo-labels
    if any(name in TREEBANKS for name in args.only):
        ru_vocab, be_vocab = load_both_vocabs()

    for name in args.only:
        if name == "trasyanka":
            records = build_trasyanka(keep_unlabeled=args.keep_unlabeled)
        else:
            records = build_treebank(name, ru_vocab, be_vocab)
        write_jsonl(records, f"{OUT_DIR}/{name}.jsonl")
        stats[name] = summarize(records)
        print(f"{name:<11} {stats[name]['sentences']:>7} sentences {stats[name]['tokens']:>9} tokens")

    # Write datasets in a fixed order, whatever order --only listed them in
    stats = {name: stats[name] for name in DATASETS if name in stats}
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
