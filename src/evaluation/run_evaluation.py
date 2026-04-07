import numpy as np
from src.features.dataset_manager import DatasetManager
from src.features.identifier import FaceIdentifier
from src.evaluation.metrics import calculate_metrics, get_confusion_matrix, generate_error_report

def run_evaluation():
    print("🚀 Démarrage de l'évaluation du système (Leave-One-Out)...")
    
    manager = DatasetManager()
    dataset = manager.load_dataset()
    
    if len(dataset) < 2:
        print("❌ Erreur : Pas assez de données dans dataset.csv pour une évaluation.")
        return

    identifier = FaceIdentifier(threshold=0.65)
    
    y_true = []
    y_pred = []
    distances = []
    
    # Leave-One-Out Cross-Validation
    for i in range(len(dataset)):
        # Target vector to evaluate
        true_name, target_vector = dataset[i]
        
        # Training set (all other vectors)
        other_vectors = dataset[:i] + dataset[i+1:]
        
        # Prediction
        pred_name, dist, is_match, conf, _ = identifier.compare(target_vector, other_vectors)
        
        y_true.append(true_name)
        # If it's a match, we use the predicted name, else "Inconnu"
        y_pred.append(pred_name if is_match else "Inconnu")
        distances.append(dist)

    # Get unique labels
    labels = sorted(list(set(y_true + y_pred)))
    
    # Calculate Metrics
    metrics = calculate_metrics(y_true, y_pred, labels)
    conf_matrix = get_confusion_matrix(y_true, y_pred, labels)
    error_report = generate_error_report(y_true, y_pred, distances)

    # Print Formatted Report
    print("\n" + "="*50)
    print("📊 RAPPORT D'ÉVALUATION ALGORITHMIQUE")
    print("="*50)
    print(f"Nombre d'échantillons testés : {len(dataset)}")
    print(f"Précision globale (Accuracy) : {metrics['accuracy']:.2%}")
    print(f"Erreur globale : {metrics['global_error']:.2%}")
    print("-"*50)
    
    print("\n📈 MÉTRIQUES PAR CLASSE :")
    for label, m in metrics['per_class'].items():
        if label == "Inconnu": continue
        print(f"\n👤 {label}:")
        print(f"   - Précision : {m['precision']:.2%}")
        print(f"   - Rappel    : {m['recall']:.2%}")
        print(f"   - Spécif.   : {m['specificity']:.2%}")
        print(f"   - TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}")

    print("\n📉 MATRICE DE CONFUSION :")
    header = "True \\ Pred | " + " | ".join([f"{l[:5]:<5}" for l in labels])
    print(header)
    print("-" * len(header))
    for t in labels:
        row = f"{t[:10]:<10} | "
        for p in labels:
            row += f"{conf_matrix[(t, p)]:<5} | "
        print(row)

    print("\n⚠️ RÉSUMÉ DES ERREURS :")
    print(error_report)
    print("\n" + "="*50)

if __name__ == "__main__":
    run_evaluation()
