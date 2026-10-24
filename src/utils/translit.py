"""
Shared Latin orthography for all three datasets.

The Trasyanka (.cha) corpus is transcribed in a Latin phonetic script, so the
Russian and Belarusian treebanks are converted *into* that script rather than
the other way around. The mapping follows how the transcripts actually spell
words, not textbook transliteration: both languages write г as "h" because
Trasyanka speakers pronounce Russian г as [ɣ] too.
"""

# Letters that turn е/ё into je/jo and и/і into ji when they come before them
CYR_VOWELS = set("аеёиоуыэюяі")
CYR_SEPARATORS = set("ьъ'’ʼ")

CYR_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d",
    "ж": "ž", "з": "z", "й": "j", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ў": "ú", "ф": "f", "х": "x", "ц": "c", "ч": "č",
    "ш": "š", "щ": "šč", "ы": "y", "э": "é", "я": "ja", "ю": "ju",
    "ь": "´", "ъ": "´´", "'": "´´", "’": "´´", "ʼ": "´´",
}

# Phonetic marks in the .cha transcripts that have no standard-spelling equivalent
CHA_STRIP = {
    "â": "a", "á": "a", "ê": "e", "ý": "y", "ô": "o", "î": "i",
    "–": "-",
}
CHA_DROP = set("<>\x9d")


def cyrillic_to_latin(word):
    """Converts a Cyrillic (Russian or Belarusian) word to the shared Latin scheme."""
    word = word.lower()
    out = []
    prev = ""
    for ch in word:
        after_vowel = prev == "" or prev in CYR_VOWELS or prev in CYR_SEPARATORS
        if ch == "е":
            out.append("je" if after_vowel else "e")
        elif ch == "ё":
            out.append("jo" if after_vowel else "ë")
        elif ch in "иі":
            # word-initial и/і stays "i"; after a vowel or ь/ъ it is [ji]
            out.append("ji" if prev and after_vowel else "i")
        else:
            out.append(CYR_TO_LAT.get(ch, ch))
        prev = ch
    return "".join(out)


def normalize_cha(word):
    """Strips transcription-only marks from a .cha word so it matches converted treebank text."""
    word = word.lower()
    return "".join(CHA_STRIP.get(ch, ch) for ch in word if ch not in CHA_DROP)
