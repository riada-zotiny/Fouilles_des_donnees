# --------------- PARTIE 2 Protocole experimental avec hyperparamètres variés
import os
import numpy as np
import pandas as pd

from time import perf_counter

from pyod.models.hbos import HBOS
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

# Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
pd.set_option('display.max_columns', None)

datasets = {
    'satellite': os.path.join(data_dir, "satellite.csv"),
    'breastcancer': os.path.join(data_dir, "breastcancer.csv"),
    'speech': os.path.join(data_dir, "speech.csv"),
    'shuttle': os.path.join(data_dir, "shuttle.csv")
}

# Dictionnaire des détecteurs avec 3 configurations d'hyperparamètres différentes
detecteurs_configs = {
    'HBOS': [
        {'n_bins': 10, 'alpha': 0.1, 'tol': 0.5},
        {'n_bins': 20, 'alpha': 0.05, 'tol': 0.3},
        {'n_bins': 15, 'alpha': 0.2, 'tol': 0.7}
    ],
    'CBLOF': [
        {'n_clusters': 8, 'alpha': 0.9, 'beta': 5},
        {'n_clusters': 10, 'alpha': 0.85, 'beta': 3},
        {'n_clusters': 12, 'alpha': 0.95, 'beta': 7}
    ],
    'IForest': [
        {'n_estimators': 100, 'max_samples': 256, 'max_features': 1.0},
        {'n_estimators': 150, 'max_samples': 128, 'max_features': 0.8},
        {'n_estimators': 200, 'max_samples': 512, 'max_features': 0.5}
    ],
    'KNN': [
        {'n_neighbors': 5, 'method': 'largest', 'metric': 'minkowski'},
        {'n_neighbors': 10, 'method': 'mean', 'metric': 'euclidean'},
        {'n_neighbors': 15, 'method': 'median', 'metric': 'manhattan'}
    ],
    'LOF': [
        {'n_neighbors': 20, 'metric': 'minkowski', 'p': 2},
        {'n_neighbors': 30, 'metric': 'euclidean', 'p': 1},
        {'n_neighbors': 40, 'metric': 'manhattan', 'p': 3}
    ],
    'OCSVM': [
        {'kernel': 'rbf', 'gamma': 'auto', 'nu': 0.5},
        {'kernel': 'linear', 'gamma': 'scale', 'nu': 0.3},
        {'kernel': 'poly', 'gamma': 'auto', 'nu': 0.7, 'degree': 3}
    ]
}

# Creation des dataFrames pour stocker les resultats
results_data = []

# Traitement de chaque jeu de données
for data_name, data_path in datasets.items(): 
    print(f"\n{'='*80}")
    print(f"Traitement du jeu de données: {data_name}")
    print(f"{'='*80}")
    
    # Charger les données 
    data = pd.read_csv(data_path).to_numpy()
    X_data = data[:, :-1]  # Features
    y_label = data[:, -1]   # Labels

    # Tester chaque détecteur avec ses 3 configurations
    for detector_name, configs in detecteurs_configs.items():
        print(f"\n--- Détecteur: {detector_name} ---")
        
        for config_idx, config in enumerate(configs, 1):
            print(f"\nConfiguration {config_idx}: {config}")
            
            # Créer le détecteur avec les hyperparamètres spécifiques
            if detector_name == 'HBOS':
                detector = HBOS(**config)
            elif detector_name == 'CBLOF':
                detector = CBLOF(**config)
            elif detector_name == 'IForest':
                detector = IForest(**config)
            elif detector_name == 'KNN':
                detector = KNN(**config)
            elif detector_name == 'LOF':
                detector = LOF(**config)
            elif detector_name == 'OCSVM':
                detector = OCSVM(**config)
            
            # Mesurer le temps d'exécution
            start = perf_counter()
            detector.fit(X_data)
            scores = detector.decision_scores_
            y_pred = detector.labels_
            end = perf_counter()
            elapsed = round(end - start, ndigits=6)
            
            # Calculer les métriques
            matrice = confusion_matrix(y_label, y_pred)
            precision = round(precision_score(y_label, y_pred), ndigits=5)
            recall = round(recall_score(y_label, y_pred), ndigits=5)
            aucroc = round(roc_auc_score(y_label, scores), ndigits=5)
            
            precisions, recalls, _ = precision_recall_curve(y_label, scores)
            aucpr = round(auc(recalls, precisions), ndigits=5)
            
            # Afficher les résultats
            print(f"Temps: {elapsed}s | Precision: {precision} | Recall: {recall} | AUC-ROC: {aucroc} | AUC-PR: {aucpr}")
            
            # Stocker les résultats
            results_data.append({
                'Dataset': data_name,
                'Detector': detector_name,
                'Config': f"Config_{config_idx}",
                'Hyperparams': str(config),
                'Precision': precision,
                'Recall': recall,
                'AUC_ROC': aucroc,
                'AUC_PR': aucpr,
                'Time': elapsed
            })

# Créer un DataFrame avec tous les résultats
results_df = pd.DataFrame(results_data)

# Afficher les résultats globaux
print(f"\n{'='*80}")
print("RÉSULTATS GLOBAUX")
print(f"{'='*80}\n")
print(results_df.to_string(index=False))

# Créer des tableaux pivot pour chaque métrique
print(f"\n{'='*80}")
print("TABLEAU RÉCAPITULATIF PAR MÉTRIQUE")
print(f"{'='*80}\n")

for metric in ['Precision', 'Recall', 'AUC_ROC', 'AUC_PR', 'Time']:
    print(f"\n--- {metric} ---")
    pivot = results_df.pivot_table(
        values=metric,
        index='Dataset',
        columns=['Detector', 'Config'],
        aggfunc='first'
    )
    print(pivot)

# Identifier les meilleures configurations pour chaque détecteur sur chaque dataset
print(f"\n{'='*80}")
print("MEILLEURES CONFIGURATIONS (par AUC-ROC)")
print(f"{'='*80}\n")

best_configs = results_df.loc[results_df.groupby(['Dataset', 'Detector'])['AUC_ROC'].idxmax()]
print(best_configs[['Dataset', 'Detector', 'Config', 'Hyperparams', 'AUC_ROC', 'Precision', 'Recall']].to_string(index=False))

# Sauvegarder les résultats dans un fichier CSV
output_path = os.path.join(current_dir, "resultats_hyperparametres.csv")
results_df.to_csv(output_path, index=False)
print(f"\n✓ Résultats sauvegardés dans: {output_path}")