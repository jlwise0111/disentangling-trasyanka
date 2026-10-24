from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np

LABELS = ["BE", "RU", "SHARED", "HYBRID"]


def flatten_sequences(y_true_seqs, y_pred_seqs):
    """
    Flattens list of sentence-level sequences into 1D lists.
    Args:
        y_true_seqs: List[List[str]] (Ground-truth tags per sentence)
        y_pred_seqs: List[List[str]] (Predicted tags per sentence)
    """
    flat_true = []
    flat_pred = []
    for true_sent, pred_sent in zip(y_true_seqs, y_pred_seqs):
        for t, p in zip(true_sent, pred_sent):
            flat_true.append(t)
            flat_pred.append(p)
    return flat_true, flat_pred


def evaluate_predictions(y_true_seqs, y_pred_seqs, labels=LABELS):
    """
    Computes token-level precision, recall, Macro-F1, and confusion matrix.
    """
    y_true, y_pred = flatten_sequences(y_true_seqs, y_pred_seqs)

    # 1. Macro-F1 score
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    # 2. Detailed Per-Class Breakdown
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        digits=4,
        zero_division=0
    )

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
        "confusion_matrix": cm,
        "labels": labels
    }


def print_evaluation_summary(metrics_dict, model_name="Model"):
    """Pretty-printer for checkpoint logs."""
    print(f"\n{'=' * 20} {model_name} Evaluation {'=' * 20}")
    print(f"Macro-F1:    {metrics_dict['macro_f1']:.4f}")
    print(f"Weighted-F1: {metrics_dict['weighted_f1']:.4f}\n")
    print("Classification Report:")
    print(metrics_dict["report"])

    print("Confusion Matrix:")
    labels = metrics_dict["labels"]
    cm = metrics_dict["confusion_matrix"]
    header = f"{'True \\ Pred':<12}" + "".join([f"{l:>10}" for l in labels])
    print(header)
    print("-" * len(header))
    for idx, row_label in enumerate(labels):
        row_str = f"{row_label:<12}" + "".join([f"{count:>10}" for count in cm[idx]])
        print(row_str)
    print("=" * (42 + len(model_name)))