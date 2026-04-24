import numpy as np
import sys
import os

# Add root project path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.features.dataset_manager import DatasetManager
from src.features.identifier import FaceIdentifier
from src.evaluation.metrics import calculate_metrics, get_confusion_matrix, generate_error_report

def run_evaluation():
    """
    Executes a Leave-One-Out (LOO) evaluation on the dataset.
    Compares each vector against all others in the dataset.
    """
    print("\n🚀 DÉMARRAGE DE L'ÉVALUATION DU SYSTÈME (Leave-One-Out)...")
    
    manager = DatasetManager()
    dataset = manager.load_dataset()
    
    if len(dataset) < 2:
        print("❌ Erreur : Pas assez de données dans dataset.csv pour une évaluation (min 2).")
        return

    # Use the same threshold as the main application
    identifier = FaceIdentifier(threshold=0.65)
    
    y_true = []
    y_pred = []
    distances = []
    
    # Process each sample in the dataset
    for i in range(len(dataset)):
        true_name, target_vector = dataset[i]
        
        # Training set = entire dataset minus the current sample (LOO)
        other_vectors = dataset[:i] + dataset[i+1:]
        
        # Perform comparison
        best_match, min_dist, is_match, confidence, top_3 = identifier.compare(target_vector, other_vectors)
        
        # y_true is the actual identity
        y_true.append(true_name)
        
        # y_pred is the predicted identity if it passes the threshold, otherwise "Inconnu"
        prediction = best_match if is_match else "Inconnu"
        y_pred.append(prediction)
        
        # Store distance for the error report
        distances.append(min_dist)

    # Compile unique labels (Actual + Predicted)
    labels = sorted(list(set(y_true + y_pred)))
    
    # Calculate Metrics
    results = calculate_metrics(y_true, y_pred, labels)
    conf_matrix = get_confusion_matrix(y_true, y_pred, labels)
    error_report = generate_error_report(y_true, y_pred, distances)

    # --- DISPLAY FORMATTED REPORT ---
    print("\n" + "="*70)
    print("📊 RAPPORT D'ÉVALUATION DE RECONNAISSANCE FACIALE")
    print("="*70)
    print(f"Échantillons totaux : {len(dataset)}")
    print(f"Classes identifiées : {len(set(y_true))}")
    print("-" * 70)
    print(f"PRÉCISION GLOBALE (ACCURACY) : {results['accuracy']:.2%}")
    print(f"ERREUR GLOBALE               : {results['global_error']:.2%}")
    print("-" * 70)
    
    print("\n📈 MÉTRIQUES PAR PERSONNE :")
    print(f"{'Nom':<15} | {'Précision':<10} | {'Rappel':<10} | {'Spécificité':<10}")
    print("-" * 55)
    
    for label in sorted(results['per_class'].keys()):
        if label == "Inconnu": continue
        m = results['per_class'][label]
        print(f"{label:<15} | {m['precision']:<10.2%} | {m['recall']:<10.2%} | {m['specificity']:<10.2%}")

    print("\n📉 MATRICE DE CONFUSION :")
    # Truncate labels for display
    disp_labels = [l[:6] for l in labels]
    header = "Vrai \\ Préd | " + " | ".join([f"{l:<6}" for l in disp_labels])
    print(header)
    print("-" * len(header))
    
    for t in labels:
        row = f"{t[:10]:<11} | "
        for p in labels:
            count = conf_matrix.get((t, p), 0)
            row += f"{count:<6} | "
        print(row)

    print("\n⚠️  DÉTAILS DES ERREURS (Leave-One-Out) :")
    print(error_report)
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    run_evaluation()
