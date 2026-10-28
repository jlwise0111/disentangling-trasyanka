"""
Feature extraction for the CRF baseline: spelling cues, prefixes and suffixes,
vocabulary membership, and the same kinds of clues for the neighbouring words.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.baseline.cues import BE_CUES, RU_CUES, BE_ENDINGS, RU_ENDINGS, has_cue, has_ending

PREFIX_LENGTHS = (1, 2, 3)
SUFFIX_LENGTHS = (1, 2, 3, 4)


def token_to_features(sent_tokens, i, ru_vocab=None, be_vocab=None):
    """
    Feature dict for the word at position i: the word, its cues, endings, vocab
    membership, prefixes and suffixes, plus a few clues from each neighbour
    (or BOS / EOS at the sentence edges).
    """
    word = sent_tokens[i]

    features = {
        "bias": 1.0,
        "word": word,
        "length": len(word),
        "has_be_cue": has_cue(word, BE_CUES),
        "has_ru_cue": has_cue(word, RU_CUES),
        "has_be_ending": has_ending(word, BE_ENDINGS),
        "has_ru_ending": has_ending(word, RU_ENDINGS),
        "in_be_vocab": be_vocab is not None and word in be_vocab,
        "in_ru_vocab": ru_vocab is not None and word in ru_vocab,
    }
    for n in PREFIX_LENGTHS:
        features[f"prefix_{n}"] = word[:n]
    for n in SUFFIX_LENGTHS:
        features[f"suffix_{n}"] = word[-n:]

    if i > 0:
        prev = sent_tokens[i - 1]
        features["-1:word"] = prev
        features["-1:suffix_2"] = prev[-2:]
        features["-1:has_be_cue"] = has_cue(prev, BE_CUES)
        features["-1:has_ru_cue"] = has_cue(prev, RU_CUES)
    else:
        features["BOS"] = True

    if i < len(sent_tokens) - 1:
        nxt = sent_tokens[i + 1]
        features["+1:word"] = nxt
        features["+1:suffix_2"] = nxt[-2:]
        features["+1:has_be_cue"] = has_cue(nxt, BE_CUES)
        features["+1:has_ru_cue"] = has_cue(nxt, RU_CUES)
    else:
        features["EOS"] = True

    return features


def sent_to_features(sent_tokens, ru_vocab=None, be_vocab=None):
    """Feature dicts for every word of one sentence."""
    return [token_to_features(sent_tokens, i, ru_vocab, be_vocab) for i in range(len(sent_tokens))]
