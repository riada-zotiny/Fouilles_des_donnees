# --------------- PARTIE 4 Protocole experimental avec des petits jeux des données 
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

from sklearn.metrics import confusion_matrix, precision_score , recall_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

# Pour afficher tout les colonnes 
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")


pd.set_option('display.max_columns', None)



datasets = {
    'satellite': os.path.join(data_dir, "satellite.csv"),
    'breastcancer': os.path.join(data_dir, "breastcancer.csv"),
    'speech': os.path.join(data_dir, "speech.csv"),
    'shuttle': os.path.join(data_dir, "shuttle.csv")
}



detecteurs = {'HBOS': HBOS(), 'CBLOF': CBLOF(), 'IForest': IForest(), 'KNN' : KNN(), 'LOF' : LOF() , 'OCSVM': OCSVM() }
N_ITERATIONS = 10



df_columns = ['Jeu de données' , 'HBOS' , 'CBLOF' , 'IForest' , 'KNN' , 'LOF' , 'OCSVM']
precision_df = pd.DataFrame(columns=df_columns)
recall_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)
aucroc_df = pd.DataFrame(columns=df_columns)
aucpr_df = pd.DataFrame(columns=df_columns)



for data_name , data_path in datasets.items(): 
    print("Data name is " , data_name , " !")
    data = pd.read_csv(data_path).to_numpy()
    drop_label = data[:, :-1] 
    X_data = drop_label 
    y_label = data[:,-1] 


    precisions_list = [data_name]
    recalls_list = [data_name]
    times_list = [data_name]
    aucroc_list = [data_name]
    aucpr_list = [data_name]

    for detector_name , detector in detecteurs.items():
        print("Detector name is " , detector_name , " !")
        precisions_iter = []
        recalls_iter = []
        times_iter = []
        aucroc_iter = []
        aucpr_iter = []
        
        for iteration in range(N_ITERATIONS):
            print(f"    Itération {iteration + 1}/{N_ITERATIONS}...", end=" ")
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
            
            precisions_iter.append(precision)
            recalls_iter.append(recall)
            times_iter.append(elapsed)
            aucroc_iter.append(aucroc)
            aucpr_iter.append(aucpr)
            
            print(f"Terminée (temps: {elapsed}s)")
       
        precision_mean = round(np.mean(precisions_iter), ndigits=5)
        recall_mean = round(np.mean(recalls_iter), ndigits=5)
        time_mean = round(np.mean(times_iter), ndigits=6)
        aucroc_mean = round(np.mean(aucroc_iter), ndigits=5)
        aucpr_mean = round(np.mean(aucpr_iter), ndigits=5)
        
        print(f"\n  Résultats moyens sur {N_ITERATIONS} itérations:")
        print(f"    Précision: {precision_mean}")
        print(f"    Rappel: {recall_mean}")
        print(f"    AUC ROC: {aucroc_mean}")
        print(f"    AUC PR: {aucpr_mean}")
        print(f"    Temps: {time_mean}s")
        
        precisions_list.append(precision_mean)
        recalls_list.append(recall_mean)
        times_list.append(time_mean)
        aucroc_list.append(aucroc_mean)
        aucpr_list.append(aucpr_mean)

    p_df = pd.DataFrame(precisions_list).transpose()
    p_df.columns = df_columns
    precision_df = pd.concat([precision_df, p_df], axis=0, ignore_index=True)

    r_df = pd.DataFrame(recalls_list).transpose()
    r_df.columns = df_columns
    recall_df = pd.concat([recall_df, r_df], axis=0, ignore_index=True)

    t_df = pd.DataFrame(times_list).transpose()
    t_df.columns = df_columns
    time_df = pd.concat([time_df, t_df], axis=0, ignore_index=True)

    roc_df = pd.DataFrame(aucroc_list).transpose()
    roc_df.columns = df_columns
    aucroc_df = pd.concat([aucroc_df, roc_df], axis=0, ignore_index=True)

    pr_df = pd.DataFrame(aucpr_list).transpose()
    pr_df.columns = df_columns
    aucpr_df = pd.concat([aucpr_df, pr_df], axis=0, ignore_index=True) 


print(f"\n{'='*70}")
print("RÉSULTATS FINAUX (Moyennes sur 10 itérations)")
print(f"{'='*70}")

print("\nPrécision\n", precision_df)
print("\nRappel\n", recall_df)
print("\nAUC ROC\n", aucroc_df)
print("\nAUC PR\n", aucpr_df)
print("\nTemps d'exécution\n", time_df)