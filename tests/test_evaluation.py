import pytest
from src.evaluation.metrics import calculate_metrics, get_confusion_matrix, generate_error_report

def test_metrics_calculation():
    y_true = ["Ahmed", "Ahmed", "Sarah", "Inconnu"]
    y_pred = ["Ahmed", "Sarah", "Sarah", "Inconnu"]
    labels = ["Ahmed", "Sarah", "Inconnu"]
    
    metrics = calculate_metrics(y_true, y_pred, labels)
    
    # Global error should be 1/4 = 0.25
    assert metrics["global_error"] == 0.25
    
    # For Ahmed: TP=1, FP=0, FN=1
    ahmed_metrics = metrics["per_class"]["Ahmed"]
    assert ahmed_metrics["precision"] == 1.0
    assert ahmed_metrics["recall"] == 0.5
    
    # For Sarah: TP=1, FP=1, FN=0
    sarah_metrics = metrics["per_class"]["Sarah"]
    assert sarah_metrics["precision"] == 0.5
    assert sarah_metrics["recall"] == 1.0

def test_confusion_matrix():
    y_true = ["A", "A", "B"]
    y_pred = ["A", "B", "B"]
    labels = ["A", "B"]
    
    matrix = get_confusion_matrix(y_true, y_pred, labels)
    assert matrix[("A", "A")] == 1
    assert matrix[("A", "B")] == 1
    assert matrix[("B", "B")] == 1
    assert matrix[("B", "A")] == 0

def test_error_report():
    y_true = ["A", "B"]
    y_pred = ["A", "C"]
    distances = [0.1, 0.6]
    
    report = generate_error_report(y_true, y_pred, distances)
    assert "Vrai='B', Prédit='C'" in report
    assert "(Distance: 0.6000)" in report
