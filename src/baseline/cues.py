"""
Spelling cues for Belarusian vs Russian in the shared Latin scheme.

Chosen from the TREEBANKS only (train splits, official Belarusian orthography), never
from Trasyanka, so the baselines stay zero-shot. Each cue below occurs in at least
98% Belarusian (or Russian) words in the treebanks.

In spoken Trasyanka several of them break down: speakers pronounce Russian words the
Belarusian way, so "čy" (Belarusian hardened ч) appears more often in RU-tagged words
(1,405) than in BE-tagged ones (595). The Russian cues "ri" and "či" hold up best.
"""

# Letters / clusters that standard spelling allows in only one of the two languages
BE_CUES = (
    "ú",     # ў
    "c´",    # ць, where Russian has ть
    "dz",    # дз / дзь, where Russian has д / дь
    "čy",    # чы: Belarusian ч is always hard
    "šy",    # шы
    "žy",    # жы
)
RU_CUES = (
    "t´",    # ть
    "ri",    # ри: Belarusian р is always hard (ры)
    "rj",    # ря / рю / рё
    "či",    # чи
    "ši",    # ши
    "ži",    # жи
)

# Word endings typical of one language (weaker evidence than the cues above)
BE_ENDINGS = ("aú", "jaú", "cca", "yja")
RU_ENDINGS = ("tsja", "t´sja", "oho", "eho", "ije")


def has_cue(word, cues):
    """True if any of the cues occurs anywhere in the word."""
    return any(cue in word for cue in cues)


def has_ending(word, endings):
    """True if the word ends with any of the endings."""
    return any(word.endswith(ending) for ending in endings)
