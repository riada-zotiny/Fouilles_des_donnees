# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import numpy as np
import pandas as pd
import os

from time import perf_counter

from pyod.models.hbos import HBOS
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

# Nombre d'itérations pour chaque détecteur
N_ITERATIONS = 10

# --------------- PARTIE 4 Protocole experimental avec des petits jeux des données mais en prennant compte leur moyenne  ------------------


current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")

pd.set_option('display.max_columns', None)

datasets = {
    'satellite': os.path.join(data_dir, "satellite.csv"),
    'breastcancer': os.path.join(data_dir, "breastcancer.csv"),
    'speech': os.path.join(data_dir, "speech.csv"),
    'shuttle': os.path.join(data_dir, "shuttle.csv")
}

detecteurs = {
    'HBOS': lambda: HBOS(), 
    'CBLOF': lambda: CBLOF(), 
    'IForest': lambda: IForest(), 
    'KNN': lambda: KNN(), 
    'LOF': lambda: LOF(), 
    'OCSVM': lambda: OCSVM()
}

# Creation des dataFrames pour stocker les resultats de chaque métrique
df_columns = ['Jeu de données', 'HBOS', 'CBLOF', 'IForest', 'KNN', 'LOF', 'OCSVM']

# DataFrames pour stocker les moyennes
precision_df = pd.DataFrame(columns=df_columns)
recall_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)
aucroc_df = pd.DataFrame(columns=df_columns)
aucpr_df = pd.DataFrame(columns=df_columns)

# DataFrames pour stocker les écarts-types
precision_std_df = pd.DataFrame(columns=df_columns)
recall_std_df = pd.DataFrame(columns=df_columns)
time_std_df = pd.DataFrame(columns=df_columns)
aucroc_std_df = pd.DataFrame(columns=df_columns)
aucpr_std_df = pd.DataFrame(columns=df_columns)

for data_name, data_path in datasets.items():
    print(f"\nTraitement de {data_name}")
    data = pd.read_csv(data_path).to_numpy()
    X_data = data[:, :-1]
    y_label = data[:, -1]

    # Listes pour stocker les résultats de chaque itération
    results = {
        'precision': {det: [] for det in detecteurs.keys()},
        'recall': {det: [] for det in detecteurs.keys()},
        'time': {det: [] for det in detecteurs.keys()},
        'aucroc': {det: [] for det in detecteurs.keys()},
        'aucpr': {det: [] for det in detecteurs.keys()}
    }

    
    for detector_name, detector_factory in detecteurs.items():
        print(f"\nExécution de {detector_name} ({N_ITERATIONS} fois)")
        
        for i in range(N_ITERATIONS):
            print(f"Itération {i+1}/{N_ITERATIONS}", end='\r')
           
            detector = detector_factory()
            
          
            start = perf_counter()
            detector.fit(X_data)
            scores = detector.decision_scores_
            y_pred = detector.labels_
            end = perf_counter()
            elapsed = round(end - start, ndigits=6)

            precision = round(precision_score(y_label, y_pred), ndigits=5)
            recall = round(recall_score(y_label, y_pred), ndigits=5)
            aucroc = round(roc_auc_score(y_label, scores), ndigits=5)
            
            precisions, recalls, _ = precision_recall_curve(y_label, scores)
            aucpr = round(auc(recalls, precisions), ndigits=5)

           
            results['precision'][detector_name].append(precision)
            results['recall'][detector_name].append(recall)
            results['time'][detector_name].append(elapsed)
            results['aucroc'][detector_name].append(aucroc)
            results['aucpr'][detector_name].append(aucpr)

        print() 


    for metric, metric_df, std_df in [
        ('precision', precision_df, precision_std_df),
        ('recall', recall_df, recall_std_df),
        ('time', time_df, time_std_df),
        ('aucroc', aucroc_df, aucroc_std_df),
        ('aucpr', aucpr_df, aucpr_std_df)
    ]:
        means = [data_name] + [np.mean(results[metric][det]) for det in detecteurs.keys()]
        stds = [data_name] + [np.std(results[metric][det]) for det in detecteurs.keys()]
        
        mean_df = pd.DataFrame([means], columns=df_columns)
        std_df_tmp = pd.DataFrame([stds], columns=df_columns)
        
        metric_df = pd.concat([metric_df, mean_df], axis=0, ignore_index=True)
        std_df = pd.concat([std_df, std_df_tmp], axis=0, ignore_index=True)

# Afficher les résultats
print("\n=== RÉSULTATS MOYENS (10 itérations) ===")
print("\nPrécision moyenne:\n", precision_df)
print("\nÉcart-type Précision:\n", precision_std_df)
print("\nRappel moyen:\n", recall_df)
print("\nÉcart-type Rappel:\n", recall_std_df)
print("\nAUC ROC moyen:\n", aucroc_df)
print("\nÉcart-type AUC ROC:\n", aucroc_std_df)
print("\nAUC PR moyen:\n", aucpr_df)
print("\nÉcart-type AUC PR:\n", aucpr_std_df)
print("\nTemps d'exécution moyen:\n", time_df)
print("\nÉcart-type Temps d'exécution:\n", time_std_df)
