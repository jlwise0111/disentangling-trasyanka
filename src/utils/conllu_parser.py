import sys
import conllu
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.translit import cyrillic_to_latin

# Letters that cannot occur in the other language's standard spelling
BE_ONLY_CHARS = set("ўіЎІ")
RU_ONLY_CHARS = set("ъищЪИЩ")

RU_FILES = [
    "data/raw/ru_syntagrus-ud-train-a.conllu",
    "data/raw/ru_syntagrus-ud-train-b.conllu",
    "data/raw/ru_syntagrus-ud-train-c.conllu",
    "data/raw/ru_syntagrus-ud-test.conllu",
]
BE_FILES = [
    "data/raw/be_hse-ud-train.conllu",
    "data/raw/be_hse-ud-test.conllu",
]
RU_VOCAB_PATH = "data/dictionaries/ru_vocab.txt"
BE_VOCAB_PATH = "data/dictionaries/be_vocab.txt"


def load_conllu_file(filename):
    # Accept either an absolute path or a path relative to PROJECT_ROOT
    path = PROJECT_ROOT / filename
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    return conllu.parse(data)


def is_word_token(token):
    """True for real words: skips multiword ranges, empty nodes, punctuation and HTML."""
    return (
            isinstance(token["id"], int)
            and token["upostag"] not in {"PUNCT", "SYM"}
            and bool(token["form"])
            and not is_html_artifact(token["form"])
    )


def build_or_load_vocab(relative_paths, cache_relative_path):
    """
    Builds a Latin-script vocabulary of word forms from raw conllu files.
    Caches it to a text file; an empty cache file is treated as missing.
    """

    cache_path = PROJECT_ROOT / cache_relative_path

    if cache_path.exists() and cache_path.stat().st_size > 0:
        print(f"Loading vocab from cache {cache_path.name}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())

    print(f"Building vocab for {cache_path.name}...")
    vocab = set()

    for path in relative_paths:
        for sentence in load_conllu_file(path):
            for token in sentence:
                if is_word_token(token):
                    vocab.add(cyrillic_to_latin(token["form"]))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for word in sorted(vocab):
            f.write(word + "\n")

    print(f"Cached {len(vocab)} unique forms to {cache_path}.")
    return vocab


def assign_language_tag(token, source_language, ru_vocab=None, be_vocab=None):
    """
    Assign a pseudo-language label to a treebank token.
    1. Letters impossible in the source language mark an embedded word
       from the other language (e.g. a Russian quote in a Belarusian text).
    2. A Latin form found in both vocabularies is SHARED.
    3. Everything else keeps the treebank's language.
    """

    if source_language == "BE" and any(c in RU_ONLY_CHARS for c in token):
        return "RU"
    if source_language == "RU" and any(c in BE_ONLY_CHARS for c in token):
        return "BE"

    latin = cyrillic_to_latin(token)
    in_be = (be_vocab is not None) and (latin in be_vocab)
    in_ru = (ru_vocab is not None) and (latin in ru_vocab)

    if in_be and in_ru:
        return "SHARED"

    return source_language


def is_html_artifact(token):
    """
    Returns True if a token appears to be HTML markup
    rather than natural-language text.
    """
    return "<" in token or ">" in token


def process_treebank(path, source_language, ru_vocab=None, be_vocab=None, with_metadata=False):
    """
    Returns one entry per sentence. By default each entry is a list of token
    dicts; with_metadata=True wraps it as {"id", "source", "split", "lang",
    "text", "tokens"}.
    """
    sentences = load_conllu_file(path)
    source = Path(path).name
    split = "test" if "-test" in source else "train"
    processed_data = []

    for sentence in sentences:
        token_list = []
        for token in sentence:
            if not is_word_token(token):
                continue

            word = token["form"]
            features = token["feats"] if token["feats"] else {}

            token_list.append({
                "form": word,
                "latin": cyrillic_to_latin(word),
                "lemma": token["lemma"],
                "upos": token["upostag"],
                "feats": "|".join(f"{k}={v}" for k, v in features.items()) or None,
                "case": features.get("Case"),
                "gender": features.get("Gender"),
                "lang_label": assign_language_tag(word, source_language, ru_vocab, be_vocab),
            })

        if not token_list:
            continue

        if with_metadata:
            processed_data.append({
                "id": sentence.metadata.get("sent_id"),
                "source": source,
                "split": split,
                "lang": source_language,
                "text": sentence.metadata.get("text"),
                "tokens": token_list,
            })
        else:
            processed_data.append(token_list)

    return processed_data


def load_both_vocabs():
    ru_vocab = build_or_load_vocab(RU_FILES, cache_relative_path=RU_VOCAB_PATH)
    be_vocab = build_or_load_vocab(BE_FILES, cache_relative_path=BE_VOCAB_PATH)
    return ru_vocab, be_vocab


if __name__ == "__main__":
    ru_vocab, be_vocab = load_both_vocabs()

    for name, files in [("RU", RU_FILES), ("BE", BE_FILES)]:
        sentences = []
        for file in files:
            sentences.extend(process_treebank(file, source_language=name, ru_vocab=ru_vocab, be_vocab=be_vocab))
        print(f"\nLoaded {len(sentences)} {name} sentences.")

        print(f"Sample {name} tokens (first 6 of sentence 2):")
        for tok in sentences[1][:6]:
            print(f"  {tok['form']:<14} {tok['latin']:<16} | UPOS: {tok['upos']:<6} | Lang: {tok['lang_label']}")

        counts = Counter(tok["lang_label"] for sent in sentences for tok in sent)
        print(f"{name} label counts: {dict(counts)}")

        for label in ["SHARED", "RU" if name == "BE" else "BE"]:
            common = Counter(tok["latin"] for sent in sentences for tok in sent if tok["lang_label"] == label)
            print(f"Most common {label} words in {name}: {[w for w, _ in common.most_common(25)]}")
