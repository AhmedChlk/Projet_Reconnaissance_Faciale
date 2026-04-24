import numpy as np
import pytest
from src.evaluation.metrics import EvaluationMetrics

def test_confusion_matrix():
    y_true = ["A", "A", "B", "C"]
    y_pred = ["A", "B", "B", "C"]
    labels = ["A", "B", "C"]
    
    matrix = EvaluationMetrics.calculate_confusion_matrix(y_true, y_pred, labels)
    
    # Expected Matrix:
    #      A  B  C (Pred)
    # A [[ 1, 1, 0 ],
    # B  [ 0, 1, 0 ],
    # C  [ 0, 0, 1 ]]
    expected = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    assert np.array_equal(matrix, expected)

def test_report_full():
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "A"]
    labels = ["A", "B"]
    
    report = EvaluationMetrics.get_report(y_true, y_pred, labels)
    
    # Each class has 1 TP, 1 FN, 1 FP.
    # Precision = 1 / (1 + 1) = 0.5
    # Recall = 1 / (1 + 1) = 0.5
    # Accuracy = 2 / 4 = 0.5
    
    assert report["A"]["precision"] == 0.5
    assert report["A"]["recall"] == 0.5
    assert report["B"]["f1_score"] == 0.5
    assert report["accuracy"] == 0.5
    assert report["A"]["support"] == 2

def test_empty_metrics():
    labels = ["A"]
    report = EvaluationMetrics.get_report([], [], labels)
    assert report["accuracy"] == 0
    assert report["A"]["precision"] == 0

def test_report_with_unknown_labels():
    y_true = ["A", "D"] # D not in labels
    y_pred = ["A", "A"]
    labels = ["A", "B"]
    
    # Should not crash, just ignore D
    report = EvaluationMetrics.get_report(y_true, y_pred, labels)
    assert report["accuracy"] == 1.0 # Only A/A was counted
    assert report["A"]["support"] == 1
