import numpy as np
from collections import Counter

def get_confusion_matrix(y_true, y_pred, labels):
    """
    Calculates a confusion matrix.
    Returns a dictionary where keys are (true_label, pred_label) and values are counts.
    """
    matrix = {(t, p): 0 for t in labels for p in labels}
    for t, p in zip(y_true, y_pred):
        if t in labels and p in labels:
            matrix[(t, p)] += 1
    return matrix

def calculate_metrics(y_true, y_pred, labels):
    """
    Calculates Precision, Recall, Specificity and Global Error for each class.
    Returns a dictionary of metrics per label.
    """
    results = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    total_samples = len(y_true)
    correct_total = np.sum(y_true == y_pred)
    global_error = 1.0 - (correct_total / total_samples) if total_samples > 0 else 0

    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        tn = np.sum((y_true != label) & (y_pred != label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results[label] = {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        }

    return {
        "per_class": results,
        "global_error": global_error,
        "accuracy": 1.0 - global_error
    }

def generate_error_report(y_true, y_pred, distances=None):
    """
    Generates a textual summary of misclassifications.
    """
    report = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p:
            dist_str = f" (Distance: {distances[i]:.4f})" if distances is not None else ""
            report.append(f"Erreur à l'index {i}: Vrai='{t}', Prédit='{p}'{dist_str}")
    
    if not report:
        return "Aucune erreur détectée."
    return "\n".join(report)
