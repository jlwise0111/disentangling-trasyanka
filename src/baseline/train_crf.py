import sys
from pathlib import Path
import sklearn_crfsuite

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.conllu_parser import process_treebank, build_or_load_vocab
from src.utils.cha_parser import load_all_cha_files
from src.utils.metrics import evaluate_predictions, print_evaluation_summary
from src.baseline.crf_features import sent_to_features

if __name__ == "__main__":
    # 1. Load Vocabularies
    ru_files = [
        "data/raw/ru_syntagrus-ud-train-a.conllu",
        "data/raw/ru_syntagrus-ud-train-b.conllu",
        "data/raw/ru_syntagrus-ud-train-c.conllu",
    ]
    be_files = [
        "data/raw/be_hse-ud-train.conllu",
        "data/raw/be_hse-ud-test.conllu",
    ]
    ru_vocab = build_or_load_vocab(ru_files, "data/dictionaries/ru_vocab.txt")
    be_vocab = build_or_load_vocab(be_files, "data/dictionaries/be_vocab.txt")

    # 2. Ingest Training Sets
    print("Loading training data from treebanks...")
    be_train = process_treebank("data/raw/be_hse-ud-train.conllu", source_language="BE", ru_vocab=ru_vocab,
                                be_vocab=be_vocab)
    ru_train_sample = process_treebank("data/raw/ru_syntagrus-ud-train-a.conllu", source_language="RU",
                                       ru_vocab=ru_vocab, be_vocab=be_vocab)[:1500]

    train_data = be_train + ru_train_sample

    X_train = [sent_to_features([tok["form"] for tok in sent], ru_vocab, be_vocab) for sent in train_data]
    y_train = [[tok["lang_label"] for tok in sent] for sent in train_data]

    # 3. Train CRF
    print(f"Training CRF on {len(X_train)} sentences...")
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)

    # 4. Evaluate across ALL Trasyanka files
    print("Evaluating CRF across all .cha files...")
    chat_data = load_all_cha_files("data/raw")

    test_tokens = [[tok["cyrillic"] for tok in utt["tokens"]] for utt in chat_data]
    y_true = [[tok["lang_label"] for tok in utt["tokens"]] for utt in chat_data]

    X_test = [sent_to_features(sent, ru_vocab, be_vocab) for sent in test_tokens]
    y_pred = crf.predict(X_test)

    # 5. Output Summary
    results = evaluate_predictions(y_true, y_pred)
    print_evaluation_summary(results, model_name="Linear-Chain CRF Baseline (Full Corpus)")