import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.metrics import evaluate_predictions, print_evaluation_summary
from src.utils.cha_parser import load_all_cha_files

# --- Orthographic Invariants ---
BE_UNIGRAMS = set("ўіЎІ")
RU_UNIGRAMS = set("ъищЪИЩ")

# Distinctive character n-grams
BE_NGRAMS = ("дзе", "дзя", "це", "ця", "цца", "аў", "яў")
RU_NGRAMS = ("тся", "ться", "ого", "его")


def load_vocab(rel_path):
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())


class NgramHeuristicClassifier:
    def __init__(self, ru_vocab_path=None, be_vocab_path=None):
        self.ru_vocab = load_vocab(ru_vocab_path) if ru_vocab_path else set()
        self.be_vocab = load_vocab(be_vocab_path) if be_vocab_path else set()

    def predict_token(self, token_text):
        token = token_text.lower()

        has_be_char = any(c in BE_UNIGRAMS for c in token)
        has_ru_char = any(c in RU_UNIGRAMS for c in token)

        # 1. Direct character contradiction within single token boundary
        if has_be_char and has_ru_char:
            return "HYBRID"

        # 2. Check n-gram morpheme clues
        has_be_ngram = any(ng in token for ng in BE_NGRAMS)
        has_ru_ngram = any(ng in token for ng in RU_NGRAMS)

        # Hybrid detection: Russian invariant char paired with a Belarusian affix
        if has_ru_char and has_be_ngram:
            return "HYBRID"
        if has_be_char and has_ru_ngram:
            return "HYBRID"

        # 3. Pure unigram invariants
        if has_be_char:
            return "BE"
        if has_ru_char:
            return "RU"

        # 4. Lexicon matching
        in_be = token in self.be_vocab
        in_ru = token in self.ru_vocab

        if in_be and in_ru:
            return "SHARED"
        if in_be and not in_ru:
            return "BE"
        if in_ru and not in_be:
            return "RU"

        # 5. N-gram fallbacks if word is out-of-vocabulary
        if has_be_ngram and not has_ru_ngram:
            return "BE"
        if has_ru_ngram and not has_be_ngram:
            return "RU"

        return "SHARED"

    def predict_sentence(self, tokens):
        """Predicts tags for a list of token strings."""
        return [self.predict_token(t) for t in tokens]

    def predict_corpus(self, sentences):
        """
        sentences: List[List[str]]
        Returns: List[List[str]]
        """
        return [self.predict_sentence(s) for s in sentences]


if __name__ == "__main__":
    # 1. Initialize classifier
    classifier = NgramHeuristicClassifier(
        ru_vocab_path="data/dictionaries/ru_vocab.txt",
        be_vocab_path="data/dictionaries/be_vocab.txt"
    )

    # 2. Ingest ALL .cha files under data/raw/
    raw_data_dir = "data/raw"
    chat_data = load_all_cha_files(raw_data_dir)

    # 3. Extract sequences
    test_tokens = [[tok["cyrillic"] for tok in utt["tokens"]] for utt in chat_data]
    y_true = [[tok["lang_label"] for tok in utt["tokens"]] for utt in chat_data]

    # 4. Predict and evaluate
    y_pred = classifier.predict_corpus(test_tokens)
    results = evaluate_predictions(y_true, y_pred)
    print_evaluation_summary(results, model_name="Rule/N-gram Heuristic Baseline (Full Corpus)")