"""
Rule-based baseline: labels each word from spelling cues, word endings and the
two treebank vocabularies. No training; the rules only use treebank knowledge.

Run from the project root:   python src/baseline/ngram_heuristic.py
Writes results/heuristic.json.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.conllu_parser import load_both_vocabs
from src.baseline.cues import BE_CUES, RU_CUES, BE_ENDINGS, RU_ENDINGS, has_cue, has_ending
from src.baseline.splits import evaluation_sets, evaluate_on_sets, print_macro_f1_table, save_results


class NgramHeuristicClassifier:
    """Labels words with fixed rules over spelling cues, endings and vocabulary lookups."""

    def __init__(self, ru_vocab, be_vocab):
        self.ru_vocab = ru_vocab
        self.be_vocab = be_vocab

    def predict_token(self, token):
        """
        Labels one lowercase Latin word; the first rule that applies wins.
        e.g. havariú -> HYBRID, tožy -> BE, ja -> SHARED, jakije -> RU, nada -> SHARED
        """
        be_cue = has_cue(token, BE_CUES)
        ru_cue = has_cue(token, RU_CUES)
        be_end = has_ending(token, BE_ENDINGS)
        ru_end = has_ending(token, RU_ENDINGS)

        # 1. cues from both languages
        if be_cue and ru_cue:
            return "HYBRID"

        # 2. a stem from one language, an ending from the other
        if (ru_cue and be_end) or (be_cue and ru_end):
            return "HYBRID"

        # 3. a cue from only one language (rules 1–2 already caught "both")
        if be_cue:
            return "BE"
        if ru_cue:
            return "RU"

        # 4. no cues at all: look the word up
        in_be = token in self.be_vocab
        in_ru = token in self.ru_vocab
        if in_be and in_ru:
            return "SHARED"
        if in_be:
            return "BE"
        if in_ru:
            return "RU"

        # 5. not in either vocabulary: fall back to endings
        if be_end and not ru_end:
            return "BE"
        if ru_end and not be_end:
            return "RU"

        # 6. nothing fired
        return "SHARED"

    def predict_sentence(self, tokens):
        """Labels a list of words."""
        return [self.predict_token(t) for t in tokens]

    def predict_corpus(self, sentences):
        """Labels a list of sentences (each a list of words)."""
        return [self.predict_sentence(s) for s in sentences]


def main():
    """Evaluates the heuristic on every evaluation set and saves results/heuristic.json."""
    ru_vocab, be_vocab = load_both_vocabs()
    classifier = NgramHeuristicClassifier(ru_vocab, be_vocab)
    results = evaluate_on_sets(classifier.predict_corpus, evaluation_sets())
    print_macro_f1_table("heuristic", results)
    print(results["trasyanka"]["report"])
    save_results("heuristic", results)


if __name__ == "__main__":
    main()
