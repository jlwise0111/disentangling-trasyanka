"""
CRF baseline: a linear-chain CRF (sklearn-crfsuite) trained on the zero-shot
training set from splits.py and evaluated on every evaluation set.

Run from the project root:   python src/baseline/train_crf.py
Takes about 2 minutes (training itself ~45 s). Writes results/crf.json.
"""
import sys
import time
from pathlib import Path

import sklearn_crfsuite

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.conllu_parser import load_both_vocabs
from src.baseline.crf_features import sent_to_features
from src.baseline.splits import (training_sentences, evaluation_sets, evaluate_on_sets,
                                 print_macro_f1_table, save_results)

CRF_PARAMS = {
    "algorithm": "lbfgs",
    "c1": 0.1,                       # L1 penalty
    "c2": 0.1,                       # L2 penalty
    "max_iterations": 100,
    "all_possible_transitions": True,
}


def main():
    """Trains the CRF on the zero-shot training set, evaluates it on every set and saves results/crf.json."""
    ru_vocab, be_vocab = load_both_vocabs()
    train = training_sentences()
    X_train = [sent_to_features(record["tokens"], ru_vocab, be_vocab) for record in train]
    y_train = [record["labels"] for record in train]

    crf = sklearn_crfsuite.CRF(**CRF_PARAMS)
    start = time.time()
    crf.fit(X_train, y_train)
    print(f"Trained in {time.time() - start:.0f}s")

    # Defined here, so it can use crf and the vocabularies; evaluate_on_sets calls it once per set
    def predict_sentences(token_lists):
        """Token lists -> predicted label lists."""
        return crf.predict([sent_to_features(tokens, ru_vocab, be_vocab) for tokens in token_lists])

    results = evaluate_on_sets(predict_sentences, evaluation_sets())
    print_macro_f1_table("crf", results)
    print(results["trasyanka"]["report"])
    save_results("crf", results)


if __name__ == "__main__":
    main()
