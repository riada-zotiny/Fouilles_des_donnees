# --------------- PARTIE 3 Protocole experimental avec des jeux des données volumineux------------
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
    'donors': os.path.join(data_dir, "donors.csv"),
    'fraud': os.path.join(data_dir, "fraud.csv"),
    'http': os.path.join(data_dir, "http.csv")
}


# Dictionnaire des détecteurs qui seront utilisés 

detecteurs = {'HBOS': HBOS(), 'CBLOF': CBLOF(), 'IForest': IForest(), 'KNN' : KNN(), 'LOF' : LOF() , 'OCSVM': OCSVM() }

# Creation des dataFrames pour stocker les resultats de chaque métrique 

df_columns = ['Jeu de données' , 'HBOS' , 'CBLOF' , 'IForest' , 'KNN' , 'LOF' , 'OCSVM']
precision_df = pd.DataFrame(columns=df_columns)
recall_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)
aucroc_df = pd.DataFrame(columns=df_columns)
aucpr_df = pd.DataFrame(columns=df_columns)

# Commencant a traiter chaque Jeu de données 

for data_name , data_path in datasets.items(): 
    print("Data name is " , data_name , " !")
    #charger les données 
    data = pd.read_csv(data_path).to_numpy()
    drop_label = data[:, :-1] # Supprimer le classifieur (Dernier colonne)
    X_data = drop_label 
    y_label = data[:,-1] # Garder uniquement la dernier colonne


    precisions_list = [data_name]
    recalls_list = [data_name]
    times_list = [data_name]
    aucroc_list = [data_name]
    aucpr_list = [data_name]

    for detector_name , detector in detecteurs.items():
        print("Detector name is " , detector_name , " !")
        start = perf_counter()
        # ajustement aux données
        detector.fit(X_data)
        # scores d'anomalies
        scores = detector.decision_scores_
        # labels prédits
        y_pred = detector.labels_
        end = perf_counter()
        elapsed = round(end - start, ndigits=6)
        print("Execution time:", elapsed)

        # matrice de confusion
        matrice = confusion_matrix(y_label, y_pred)
        print(matrice)
        
        # précision et rappel (classe positive, i.e., les anomalies)
        precision = round(precision_score(y_label, y_pred), ndigits=5)
        recall = round(recall_score(y_label, y_pred), ndigits=5)
         
        # aire sous la courbe ROC
        aucroc = round(roc_auc_score(y_label, scores), ndigits=5)
        
        # aire sous la courbe PR
        precisions, recalls, _ = precision_recall_curve(y_label, scores)
        aucpr = round(auc(recalls, precisions), ndigits=5)
        
        print(f"Precision:{precision} \nRecall:{recall}")
        print(f"AUCROC:{aucroc}")
        print(f"AUCPR:{aucpr}")

        precisions_list.append(precision)
        recalls_list.append(recall)
        times_list.append(elapsed)
        aucroc_list.append(aucroc)
        aucpr_list.append(aucpr)

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

    print("\n------------------------------------------------------------------")


print("Précision\n", precision_df)
print("\nRappel\n", recall_df)
print("\nAUC ROC\n", aucroc_df)
print("\nAUC PR\n", aucpr_df)
print("\nTemps d'exécution\n", time_df)