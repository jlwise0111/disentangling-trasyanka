"""
Parser for the Trasyanka corpus (CHAT .cha transcripts).

Each utterance is a speech line followed by its %mor line:
    *AFV:   ničohâ@b@pb mne@c@pc ...
    %mor:   b:pb#pro:neg|_ništo_ničto_&GEN-N:SG c:pc#pro:pers:dei|_ja_ja_&DAT-1:SG ...

Every word gets its gold language label (BE / RU / SHARED / HYBRID), its phonetic
label, a normalised Latin form shared with the treebanks, and the POS, lemmas and
features from the aligned %mor entry.
"""
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.translit import normalize_cha

# Word-level language codes (@b, @r, ...) and phonetic codes (@pb, @pr, ...)
LANG_CODES = {"b": "BE", "r": "RU", "c": "SHARED", "h": "HYBRID"}
PHON_CODES = {"pb": "BE", "pr": "RU", "pc": "SHARED", "ph": "HYBRID"}

# Anonymised words: they have a %mor entry but carry no language evidence
PLACEHOLDERS = {"persname", "swearword"}
# Unintelligible speech, written xx@u; it has NO %mor entry
UNINTELLIGIBLE = "xx"

PUNCTUATION = {".", ",", "?", "!", ":", ";"}

# <word@lang@phon@flag...>[annotation]
#   ^<?           optional opening retracing bracket
#   ([^@<>]+)     group 1: the word itself (anything up to the first @)
#   @([a-z])      group 2: one-letter language code (b r c h, or u k)
#   (?:@(p[a-z]))?  group 3: optional phonetic code (pb pr pc ph)
#   ((?:@[a-z]+)*)  group 4: ALL remaining flags as one string, e.g. "@w@dial"
#   >?            optional closing retracing bracket; anything after it is ignored
TOKEN_RE = re.compile(r"^<?([^@<>]+)@([a-z])(?:@(p[a-z]))?((?:@[a-z]+)*)>?")


def is_skippable(raw_tok):
    """
    True for tokens that are not words: punctuation, pauses (.), annotations [..],
    terminators +..., omitted 0words and &fragments. Used for both the speech and %mor lines.
    """
    if raw_tok in PUNCTUATION or raw_tok.startswith(("(", "[", "+", "0", "&")):
        return True

    return False


def parse_token(raw_tok):
    """
    Splits a speech-line token such as "<atrézâú@h@pb@w>" into
    {"raw", "latin", "lang_label", "phon_label", "flags"}.

    Unclear-language (@u) words, fragments (@k) and placeholders get labels None.
    Returns None for anything that is not a word.
    """
    parts = TOKEN_RE.search(raw_tok)
    if not parts:
        # Untagged token: only anonymised placeholders count as words (they have a %mor entry)
        word = raw_tok.strip("<>").split(">")[0].split("[")[0]
        if word in PLACEHOLDERS:
            return {"raw": word, "latin": word, "lang_label": None, "phon_label": None, "flags": []}
        return None

    word = parts.group(1)
    lang = parts.group(2)
    phon = parts.group(3)
    flags = parts.group(4).split("@")
    flags = [f for f in flags if f]

    # Placeholders carry no language evidence, even when tagged (persname@c@pc)
    if word in PLACEHOLDERS:
        lang = None
        phon = None

    return {
        "raw": word,
        "latin": normalize_cha(word),
        "lang_label": LANG_CODES.get(lang),
        "phon_label": PHON_CODES.get(phon),
        "flags": flags,
    }


def parse_mor_entry(entry):
    """
    Parses one %mor item, e.g. "c:pb#v:ipf|_havarycca_govorit´sja_&PRAES-3:SG",
    into {"pos", "be_lemma", "ru_lemma", "feats"}. A "0" lemma or a missing part becomes None.
    """
    head, _, rest = entry.partition("|")
    pos = head.partition("#")[2]
    lemma_part, _, feats = rest.partition("&")

    lemmas = lemma_part.strip("_").split("_")
    be_lemma = lemmas[0]
    ru_lemma = lemmas[1] if len(lemmas) > 1 else None

    if be_lemma == "0":
        be_lemma = None
    if ru_lemma == "0":
        ru_lemma = None

    return {"pos": pos, "be_lemma": be_lemma, "ru_lemma": ru_lemma, "feats": feats or None}


def parse_cha_file(filename_or_path, keep_unlabeled=False):
    """
    Parses one .cha file into a list of utterances:
    {"id", "source", "family", "speaker", "mor_aligned", "tokens"}.

    Words without one of the four labels are dropped unless keep_unlabeled=True.
    """
    # Accept either an absolute Path/str or a relative path from PROJECT_ROOT
    path = Path(filename_or_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / filename_or_path

    if not path.exists():
        raise FileNotFoundError(f"Cannot find CHAT file at: {path}")

    # utf-8-sig strips the byte-order mark at the start of each file
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # "fam_ak_1" -> "fam_ak"
    family_name = re.sub(r"_\d+$", "", path.stem)

    dataset = []

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("*"):
            speaker_match = re.match(r"\*([A-Z0-9]+):\s+(.*)", line)
            if not speaker_match:
                continue

            speaker, utterance = speaker_match.groups()
            raw_tokens = utterance.split()

            # Keep unlabeled words for now: they still have a %mor entry
            sentence_tokens = []
            for raw_tok in raw_tokens:
                if is_skippable(raw_tok):
                    continue
                token = parse_token(raw_tok)

                if token and token["raw"] != UNINTELLIGIBLE:
                    sentence_tokens.append(token)

            # The %mor line has one entry per word, in the same order
            mor_line = lines[i + 1] if i + 1 < len(lines) else ""
            mor_entries = []
            if mor_line.startswith("%mor"):
                parts = mor_line.split("\t", 1)[1].split()
                mor_entries = [part for part in parts if not is_skippable(part)]

            # On a count mismatch, don't guess: the grammar fields stay None
            mor_aligned = len(mor_entries) == len(sentence_tokens)

            for j, tok in enumerate(sentence_tokens):
                mor = parse_mor_entry(mor_entries[j]) if mor_aligned else {}
                for key in ("pos", "be_lemma", "ru_lemma", "feats"):
                    tok[key] = mor.get(key)

            # Only now drop unlabeled words, so the alignment above stays correct
            if not keep_unlabeled:
                sentence_tokens = [token for token in sentence_tokens if token["lang_label"] is not None]

            if sentence_tokens:
                dataset.append({
                    "id": f"{path.name}:{i + 1}",
                    "source": path.name,
                    "family": family_name,
                    "speaker": speaker,
                    "mor_aligned": mor_aligned,
                    "tokens": sentence_tokens
                })

    return dataset


def load_all_cha_files(directory_path, keep_unlabeled=False):
    """
    Finds and parses all .cha files in a directory or its subdirectories.
    Returns a unified list of utterance dictionaries.
    """
    dir_path = Path(directory_path)
    if not dir_path.is_absolute():
        dir_path = PROJECT_ROOT / directory_path

    cha_files = sorted(dir_path.glob("**/*.cha"))
    if not cha_files:
        raise FileNotFoundError(f"No .cha files found in: {dir_path}")

    all_utterances = []
    total_tokens = 0

    print(f"Discovered {len(cha_files)} .cha files in {dir_path}...")
    for file_path in cha_files:
        utterances = parse_cha_file(file_path, keep_unlabeled=keep_unlabeled)
        all_utterances.extend(utterances)
        for utt in utterances:
            total_tokens += len(utt["tokens"])

    print(f"Loaded {len(all_utterances)} total utterances ({total_tokens} tokens) across all files.\n")
    return all_utterances


if __name__ == "__main__":
    # Point directly to the file inside data/raw/
    sample_file = "data/raw/fam_ak_1.cha"

    data = parse_cha_file(sample_file)

    print(f"Extracted {len(data)} Trasyanka utterances!\n")
    if data:
        print("Sample parsed utterance:")
        for tok in data[0]["tokens"]:
            print(f"  {tok['raw']:<15} -> {tok['latin']:<12} [{tok['lang_label']:<6} phon={str(tok['phon_label']):<6}] "
                  f"{str(tok['pos']):<14} BE={tok['be_lemma']} RU={tok['ru_lemma']}")
