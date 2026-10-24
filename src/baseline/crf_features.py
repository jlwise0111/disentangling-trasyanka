import sys
from pathlib import Path

BE_UNIGRAMS = set("ўіЎІ")
RU_UNIGRAMS = set("ъищЪИЩ")


def token_to_features(sent_tokens, i, ru_vocab=None, be_vocab=None):
    """
    Extracts rich orthographic, affix, and sequential context features
    for token at index i in sent_tokens.
    """
    word = sent_tokens[i]
    word_lower = word.lower()

    features = {
        "bias": 1.0,
        "word.lower()": word_lower,
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "length": len(word),

        # Character-level presence indicators
        "has_be_char": any(c in BE_UNIGRAMS for c in word),
        "has_ru_char": any(c in RU_UNIGRAMS for c in word),

        # Affix features (lengths 1, 2, 3, 4)
        "prefix_1": word_lower[:1],
        "prefix_2": word_lower[:2] if len(word_lower) >= 2 else "",
        "prefix_3": word_lower[:3] if len(word_lower) >= 3 else "",
        "suffix_1": word_lower[-1:],
        "suffix_2": word_lower[-2:] if len(word_lower) >= 2 else "",
        "suffix_3": word_lower[-3:] if len(word_lower) >= 3 else "",
        "suffix_4": word_lower[-4:] if len(word_lower) >= 4 else "",

        # Lexicon lookup indicators
        "in_be_vocab": (be_vocab is not None) and (word_lower in be_vocab),
        "in_ru_vocab": (ru_vocab is not None) and (word_lower in ru_vocab),
    }

    # Context feature: Previous token (w_{i-1})
    if i > 0:
        prev_word = sent_tokens[i - 1]
        features.update({
            "-1:word.lower()": prev_word.lower(),
            "-1:has_be_char": any(c in BE_UNIGRAMS for c in prev_word),
            "-1:has_ru_char": any(c in RU_UNIGRAMS for c in prev_word),
            "-1:suffix_2": prev_word.lower()[-2:] if len(prev_word) >= 2 else "",
        })
    else:
        features["BOS"] = True  # Beginning of sentence

    # Context feature: Next token (w_{i+1})
    if i < len(sent_tokens) - 1:
        next_word = sent_tokens[i + 1]
        features.update({
            "+1:word.lower()": next_word.lower(),
            "+1:has_be_char": any(c in BE_UNIGRAMS for c in next_word),
            "+1:has_ru_char": any(c in RU_UNIGRAMS for c in next_word),
            "+1:suffix_2": next_word.lower()[-2:] if len(next_word) >= 2 else "",
        })
    else:
        features["EOS"] = True  # End of sentence

    return features


def sent_to_features(sent_tokens, ru_vocab=None, be_vocab=None):
    return [token_to_features(sent_tokens, i, ru_vocab, be_vocab) for i in range(len(sent_tokens))]