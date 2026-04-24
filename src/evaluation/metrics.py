import numpy as np
from typing import List, Dict, Tuple, Set

class EvaluationMetrics:
    @staticmethod
    def calculate_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> np.ndarray:
        """Calculates a confusion matrix."""
        n = len(labels)
        label_to_idx = {label: i for i, label in enumerate(labels)}
        matrix = np.zeros((n, n), dtype=int)
        
        for true, pred in zip(y_true, y_pred):
            if true in label_to_idx and pred in label_to_idx:
                matrix[label_to_idx[true], label_to_idx[pred]] += 1
        return matrix

    @staticmethod
    def get_report(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
        """Generates a full evaluation report including accuracy, precision, recall, and f1."""
        if not y_true or not y_pred:
            report = {label: {"precision": 0, "recall": 0, "f1_score": 0, "support": 0} for label in labels}
            report["accuracy"] = 0
            return report

        matrix = EvaluationMetrics.calculate_confusion_matrix(y_true, y_pred, labels)
        report = {}
        
        for i, label in enumerate(labels):
            tp = matrix[i, i]
            fp = matrix[:, i].sum() - tp
            fn = matrix[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            report[label] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": int(tp + fn)
            }
            
        report["accuracy"] = np.trace(matrix) / np.sum(matrix) if np.sum(matrix) > 0 else 0
        return report

# Free functions for run_evaluation.py script
def calculate_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    """
    Calculates precision, recall, specificity, and accuracy for each label.
    Used by run_evaluation.py.
    """
    per_class = {}
    total = len(y_true)
    
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t != label and p != label)
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "specificity": specificity
        }
        
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "global_error": 1.0 - accuracy,
        "per_class": per_class
    }

def get_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[Tuple[str, str], int]:
    """
    Generates a confusion matrix as a dictionary for easy display in script.
    """
    matrix = {}
    for t in labels:
        for p in labels:
            matrix[(t, p)] = sum(1 for true, pred in zip(y_true, y_pred) if true == t and pred == p)
    return matrix

def generate_error_report(y_true: List[str], y_pred: List[str], distances: List[float]) -> str:
    """
    Summarizes misclassifications and analysis of distances.
    """
    errors = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p:
            errors.append(f"Erreur : Vrai='{t}', Prédit='{p}' (Distance min : {distances[i]:.3f})")
    
    if not errors:
        return "✅ Aucune erreur détectée sur cet ensemble de données."
    
    return "\n".join(errors)
