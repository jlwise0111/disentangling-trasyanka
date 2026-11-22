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

# ---------------------------------------------------------------------------
# Latin -> Cyrillic, for the pretrained models (fastText, SlavicBERT, XLM-R).
#
# The output never depends on the word's language label: the label is what the
# models predict, so using it to build their input would leak the answer.
# Every word is written in Belarusian spelling, which follows pronunciation the
# way the transcripts do (i -> і, ú -> ў, ´´ -> '). Spoken Russian therefore comes
# out as Belarusian-looking Cyrillic (nada -> нада, tožy -> тожы).
# This matches the old converter's output (data/old_processed/) except for ´´,
# which the old converter wrote as "ьь".
# ---------------------------------------------------------------------------

# Two-letter Latin sequences that become one Cyrillic letter. Check these before
# single letters, so "ja" becomes я and not й + а.
LAT_SEQUENCES = {
    "´´": "'",      # separating apostrophe: p´´ju -> п'ю
    "šč": "щ",      # ščas -> щас
    "ja": "я",      # also after consonants: mjane -> мяне, dlja -> для
    "je": "е",
    "jé": "е",
    "jo": "ё",
    "jë": "ё",      # žyl´jë -> жыльё
    "ju": "ю",
    "ji": "і",      # jix -> іх
}

# Single Latin letters. g is г too: the transcripts write Russian г as g in a few
# words (kagda, gde), and ґ is too rare in real text to help a pretrained model.
LAT_TO_CYR = {
    "a": "а", "b": "б", "v": "в", "h": "г", "g": "г", "d": "д",
    "ž": "ж", "z": "з", "j": "й", "k": "к", "l": "л", "m": "м",
    "n": "н", "o": "о", "p": "п", "r": "р", "s": "с", "t": "т",
    "u": "у", "ú": "ў", "f": "ф", "x": "х", "c": "ц", "č": "ч",
    "š": "ш", "y": "ы", "é": "э", "e": "е", "ë": "ё", "i": "і",
    "´": "ь",
}


def latin_to_cyrillic(word):
    """
    Converts a word in the shared Latin scheme (a "tokens" entry) to Belarusian
    Cyrillic. Characters in neither table (- and _) are kept as they are.
        havaryú -> гаварыў    mjane -> мяне    p´´ju -> п'ю    ščas -> щас
    """
    cyrillic = ""
    i = 0
    while i < len(word):
        if word[i : i + 2] in LAT_SEQUENCES:
            cyrillic += LAT_SEQUENCES[word[i : i + 2]]
            i += 2
        elif word[i] in LAT_TO_CYR:
            cyrillic += LAT_TO_CYR[word[i]]
            i += 1
        else:
            cyrillic += word[i]
            i += 1

    return cyrillic


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
