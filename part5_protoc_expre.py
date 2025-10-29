# --------------- PARTIE 4 Protocole experimental avec 10 itérations ------------------

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

# Dictionnaire des détecteurs
detecteurs = {
    'HBOS': HBOS,
    'CBLOF': CBLOF,
    'IForest': IForest,
    'KNN': KNN,
    'LOF': LOF,
    'OCSVM': OCSVM
}

# Nombre d'itérations
n_iterations = 10

# Initialisation des DataFrames pour les résultats
metriques = ['Precision', 'Recall', 'AUC_ROC', 'AUC_PR', 'Time']

# Dictionnaire pour stocker tous les résultats
all_results = {metric: [] for metric in metriques}

# Traitement de chaque jeu de données
for data_name, data_path in datasets.items(): 
    print(f"\nTraitement de {data_name}")
    
    # Charger les données 
    data = pd.read_csv(data_path).to_numpy()
    X_data = data[:, :-1]  # Features
    y_label = data[:, -1]   # Labels

    # Dictionnaire pour stocker les résultats de ce dataset
    dataset_results = {
        'Precision': {},
        'Recall': {},
        'AUC_ROC': {},
        'AUC_PR': {},
        'Time': {}
    }

    # Tester chaque détecteur
    for detector_name, DetectorClass in detecteurs.items():
        print(f"\nExécution de {detector_name} ({n_iterations} fois)")
        
        # Listes pour stocker les résultats de chaque itération
        precisions = []
        recalls = []
        aucrocs = []
        aucprs = []
        times = []
        
        # Exécuter n_iterations fois
        for iteration in range(n_iterations):
            print(f"\rItération {iteration + 1}/{n_iterations}", end='')
            
            # Créer et entraîner le détecteur
            detector = DetectorClass()
            
            start = perf_counter()
            detector.fit(X_data)
            scores = detector.decision_scores_
            y_pred = detector.labels_
            end = perf_counter()
            elapsed = round(end - start, ndigits=6)
            
            # Calculer les métriques
            precision = round(precision_score(y_label, y_pred), ndigits=5)
            recall = round(recall_score(y_label, y_pred), ndigits=5)
            aucroc = round(roc_auc_score(y_label, scores), ndigits=5)
            
            precisions_curve, recalls_curve, _ = precision_recall_curve(y_label, scores)
            aucpr = round(auc(recalls_curve, precisions_curve), ndigits=5)
            
            # Stocker les résultats
            precisions.append(precision)
            recalls.append(recall)
            aucrocs.append(aucroc)
            aucprs.append(aucpr)
            times.append(elapsed)
        
        print()  # Nouvelle ligne après les itérations
        
        # Calculer les moyennes et écarts-types
        dataset_results['Precision'][detector_name] = {
            'mean': np.mean(precisions),
            'std': np.std(precisions)
        }
        dataset_results['Recall'][detector_name] = {
            'mean': np.mean(recalls),
            'std': np.std(recalls)
        }
        dataset_results['AUC_ROC'][detector_name] = {
            'mean': np.mean(aucrocs),
            'std': np.std(aucrocs)
        }
        dataset_results['AUC_PR'][detector_name] = {
            'mean': np.mean(aucprs),
            'std': np.std(aucprs)
        }
        dataset_results['Time'][detector_name] = {
            'mean': np.mean(times),
            'std': np.std(times)
        }
    
    # Ajouter les résultats de ce dataset aux résultats globaux
    for metric in metriques:
        row_mean = {'Jeu de données': data_name}
        row_std = {'Jeu de données': data_name}
        
        for detector_name in detecteurs.keys():
            row_mean[detector_name] = round(dataset_results[metric][detector_name]['mean'], 5)
            row_std[detector_name] = round(dataset_results[metric][detector_name]['std'], 5)
        
        all_results[metric].append({'mean': row_mean, 'std': row_std})

# Créer les DataFrames finaux
print("\n" + "="*80)
print("=== RÉSULTATS MOYENS (10 itérations) ===")
print("="*80 + "\n")

for metric in metriques:
    mean_data = [item['mean'] for item in all_results[metric]]
    std_data = [item['std'] for item in all_results[metric]]
    
    mean_df = pd.DataFrame(mean_data)
    std_df = pd.DataFrame(std_data)
    
    print(f"{metric} moyen:")
    print(mean_df.to_string(index=False))
    print(f"\nÉcart-type {metric}:")
    print(std_df.to_string(index=False))
    print("\n" + "-"*80 + "\n")

# Sauvegarder les résultats dans un fichier CSV
output_path = os.path.join(current_dir, "resultats_10iterations.csv")

# Créer un DataFrame complet avec toutes les informations
final_results = []
for data_name in datasets.keys():
    for detector_name in detecteurs.keys():
        row = {'Dataset': data_name, 'Detector': detector_name}
        for metric in metriques:
            # Trouver les résultats pour ce dataset
            dataset_idx = list(datasets.keys()).index(data_name)
            row[f'{metric}_mean'] = all_results[metric][dataset_idx]['mean'][detector_name]
            row[f'{metric}_std'] = all_results[metric][dataset_idx]['std'][detector_name]
        final_results.append(row)

final_df = pd.DataFrame(final_results)
final_df.to_csv(output_path, index=False)
print(f"✓ Résultats sauvegardés dans: {output_path}")