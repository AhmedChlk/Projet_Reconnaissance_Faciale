# GEMINI.MD : Spécifications Techniques et Standards de Développement

## 1. Contexte et Vision du Projet
Ce projet consiste à concevoir et implémenter un système complet d'identification faciale basé exclusivement sur des techniques classiques de vision par ordinateur **Il est strictement interdit de recourir à l'apprentissage automatique ou au deep learning** L'approche adoptée repose sur trois modules principaux : la détection/normalisation, le contour actif (Snake), et les descripteurs géométriques10, 11, . L'objectif est de produire un logiciel industriel, robuste, et maintenable.

## 2. Architecture du Système
Le système se décompose en deux modes d'utilisation distincts:

### A. Mode Enregistrement (Offline)
1. Capture d'une photo de la personne.
2. Cadrage Haar et alignement.
3. Algorithme Snake pour extraire le contour ovale facial.
4. Extraction des points caractéristiques et calcul des distances.
5. Sauvegarde d'un vecteur de 30 dimensions (30D) et du nom de la personne dans le fichier `dataset.csv` .

### B. Mode Identification (En Ligne)
1. Capture d'une photo inconnue.
2. Cadrage Haar et alignement.
3. Algorithme Snake pour le contour ovale facial.
4. Extraction des points caractéristiques et distances.
5. Comparaison du vecteur obtenu à **tous** les vecteurs du dataset .
6. Identification basée sur la plus petite distance : Personne X ou Inconnu .

---

## 3. Spécifications par Module

### Module 1 : Création du Dataset
* **Contenu** : Minimum 10 personnes, avec un minimum de 20 images par personne.
* **Variations** : Visage de face (neutre), avec expressions (sourire, surprise), rotations (+/- 15 degrés), et éclairages différents  .
* **Format** : Les images sont en niveaux de gris et redimensionnées à $128\times128$ pixels après cadrage.
* **Stockage** : Fichier CSV (`dataset.csv`). Chaque ligne contient le nom et les 30 valeurs séparées par des points-virgules. **Aucune image n'est stockée, seul le vecteur est conservé**.

### Module 2 : Détection et Alignement
Cette étape est critique pour la suite du pipeline.
1. **Détection Haar** : Localiser le visage pour obtenir la boîte englobante.
2. **Sélection** : Prendre le plus grand visage si plusieurs sont détectés.
3. **Détection des yeux** : Localiser l'œil gauche et l'œil droit.
4. **Calcul de l'angle** : `angle = arctan((cy_d - cy_g) / (cx_d - cx_g))`.
5. **Rotation** : Aligner les yeux horizontalement.
6. **Normalisation** : Redimensionnement à $128\times128$ et égalisation d'histogramme pour lisser la luminosité.

### Module 3 : Contour Actif et Points Caractéristiques
* **Snake** : Appliquer le modèle de contour actif sur le visage normalisé pour extraire le contour de l'ovale facial. Ce contour sert de descripteur de forme.
* **Biométrie** : Extraire les points géométriques clés (yeux, nez, bouche) et calculer les distances inter-points normalisées. Ces distances forment la signature biométrique de la personne.

### Module 4 : Moteur de Recherche et Décision
* **Comparaison** : Recherche directe par distance euclidienne. Aucun entraînement n'est nécessaire.
* **Score de confiance** : Une distance de $0$ donne $100\%$ de confiance. Une distance proche du seuil donne une confiance proche de $0\%$ .
* **Seuil** : Hyperparamètre crucial à déterminer empiriquement sur le dataset .
* **Top 3** : Le système doit afficher les 3 candidats les plus proches avec leurs distances pour faciliter le débogage .

### Module 5 : Simulation de Porte Sécurisée (IHM)
Il s'agit du module de sortie visible.
* **Interface Tkinter** : Fenêtre animant une porte en 3D avec voyant vert ou rouge.
* **Overlay OpenCV** : Superposition sur le flux webcam affichant le nom, la distance, le score de confiance et une icône.
* **Logique** : 
  * `distance <= seuil` -> Accès autorisé (Nom affiché, porte ouverte, voyant vert).
  * `distance > seuil` -> Accès refusé (Inconnu, porte fermée, voyant rouge clignotant).
* **Commandes** : `I` (Lancer l'identification), `E` (Enregistrer), `Q` (Quitter).

### Module 6 : Évaluation
Évaluer le système via la matrice de confusion, la précision, le rappel, la spécificité et l'erreur. Analyser les confusions pour comprendre si elles proviennent du vecteur Snake ou des distances.

---

## 4. Standards d'Ingénierie et "Clean Code"
Pour garantir un code de niveau professionnel, complet, et sans aucune "trace d'IA" générique :
* **Architecture Modulaire (SOLID)** : Chaque étape du pipeline (Capture, Prétraitement, Snake, Extraction, Persistance, Interface) doit être isolée dans son propre module ou classe.
* **Aucun Code Smell** : Pas de variables globales, pas de fonctions de plus de 20 lignes, et strict respect du principe DRY (Don't Repeat Yourself).
* **Conventions de Nommage (PEP 8)** : Noms de variables et de fonctions métiers explicites (ex: `extract_facial_contour` plutôt que `do_snake`).
* **Typage Statique** : Utilisation exhaustive des "Type Hints" Python (`-> np.ndarray`, `-> float`, etc.).
* **Documentation** : Docstrings (format Google ou NumPy) sur toutes les méthodes publiques pour expliquer le "Pourquoi" (le "Comment" doit être évident à la lecture du code).

## 5. Stratégie de Test Rigoureuse
Le projet doit être garanti "Zero Bug" grâce à une couverture de tests exhaustive (framework recommandé : `pytest`) :
* **Tests Unitaires** : Tester mathématiquement chaque fonction géométrique (calcul d'angle de rotation, distance euclidienne, convergence du Snake sur une forme connue).
* **Tests d'Intégration** : Validation du flux de données complet, de la capture virtuelle à la génération du vecteur 30D.
* **Mocking** : La logique de la porte (Tkinter) et la lecture webcam (OpenCV) doivent pouvoir être "mockées" pour tester le cœur de l'application en intégration continue (CI).

## 6. Livrables Finaux
* Code source complet, modulaire et testé.
* Fichier `dataset.csv`.
* Rapport technique incluant l'évaluation des performances.
* Présentation du projet.