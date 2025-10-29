
import pandas as pd
import os

# Charger les données

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")

df_satellite = pd.read_csv(os.path.join(data_dir, "satellite.csv"))
df_breastcancer = pd.read_csv(os.path.join(data_dir, "breastcancer.csv"))
df_speech = pd.read_csv(os.path.join(data_dir, "speech.csv"))
df_shuttle = pd.read_csv(os.path.join(data_dir, "shuttle.csv"))
df_fraud = pd.read_csv(os.path.join(data_dir, "fraud.csv"))
df_donors = pd.read_csv(os.path.join(data_dir, "donors.csv"))
df_http = pd.read_csv(os.path.join(data_dir, "http.csv"))
# Satellite
print(df_satellite.shape)

nbs_observation_satellite , nbs_variables_satellite = df_satellite.shape

print(df_satellite.columns)

dect_anomalies_satellite = (df_satellite["36"] == 1 ).sum() 

perc_dect_anomalies_satellite = (dect_anomalies_satellite / nbs_observation_satellite) * 100 

print(dect_anomalies_satellite , perc_dect_anomalies_satellite)


# Breastcancer
nbs_observation_breastcancer , nbs_variables_breastcancer = df_breastcancer.shape 

dect_anomalies_breastcancer = (df_breastcancer["30"] == 1).sum() 

perc_dect_anomalies_breastcancer = (dect_anomalies_breastcancer / nbs_observation_breastcancer) * 100 

print(dect_anomalies_breastcancer , perc_dect_anomalies_breastcancer)

#Speech

nbs_observation_speech , nbs_variables_speech = df_speech.shape

print(df_speech.columns)
print(df_speech["400"].value_counts())

dect_anomalies_speech = (df_speech["400"] == 1).sum()

perc_dect_anomalies_speech = (dect_anomalies_speech / nbs_observation_speech) * 100 

print(dect_anomalies_speech , perc_dect_anomalies_speech)


#Shuttle

nbs_observation_shuttle , nbs_variables_shuttle = df_shuttle.shape
print(df_shuttle.columns)

dect_anomalies_shuttle = (df_shuttle["9"] == 1).sum()

perc_dect_anomalies_shuttle = (dect_anomalies_shuttle / nbs_observation_shuttle) * 100 

print(dect_anomalies_shuttle , perc_dect_anomalies_shuttle)

# Fraud

nbs_observation_fraud , nbs_variables_fraud = df_fraud.shape
print(df_fraud.columns)

dect_anomalies_fraud = (df_fraud["class"] == 1).sum()

perc_dect_anomalies_fraud = (dect_anomalies_fraud / nbs_observation_fraud) * 100 

print(dect_anomalies_fraud , perc_dect_anomalies_fraud)

# Donors

nbs_observation_donors , nbs_variables_donors = df_donors.shape
print(df_donors.columns)

dect_anomalies_donors = (df_donors["class"] == 1).sum()

perc_dect_anomalies_donors = (dect_anomalies_donors / nbs_observation_donors) * 100 

print(dect_anomalies_donors , perc_dect_anomalies_donors)

# HTTP

nbs_observation_http , nbs_variables_http = df_http.shape
print(df_http.columns)

dect_anomalies_http = (df_http["29"] == 1).sum()

perc_dect_anomalies_http = (dect_anomalies_http / nbs_observation_http) * 100 
print(dect_anomalies_http , perc_dect_anomalies_http)


from tabulate import tabulate

# Création des données pour le tableau
headers = ["Jeu de données", "Nbre d'observations", "Nbre de variables", "Nbre d'anomalies", "Pourcentage d'anomalies"]
table_data = [
    ["Satellite", nbs_observation_satellite, nbs_variables_satellite, dect_anomalies_satellite, f"{perc_dect_anomalies_satellite:.2f}%"],
    ["Breastcancer", nbs_observation_breastcancer, nbs_variables_breastcancer, dect_anomalies_breastcancer, f"{perc_dect_anomalies_breastcancer:.2f}%"],
    ["Speech", nbs_observation_speech, nbs_variables_speech, dect_anomalies_speech, f"{perc_dect_anomalies_speech:.2f}%"],
    ["Shuttle", nbs_observation_shuttle, nbs_variables_shuttle, dect_anomalies_shuttle, f"{perc_dect_anomalies_shuttle:.2f}%"],
    ["Fraud", nbs_observation_fraud, nbs_variables_fraud, dect_anomalies_fraud, f"{perc_dect_anomalies_fraud:.2f}%"],
    ["Donors", nbs_observation_donors, nbs_variables_donors, dect_anomalies_donors, f"{perc_dect_anomalies_donors:.2f}%"],
    ["Http", nbs_observation_http, nbs_variables_http, dect_anomalies_http, f"{perc_dect_anomalies_http:.2f}%"]
]

# Affichage du tableau avec tabulate
print("\nRésultats finaux:")
print(tabulate(table_data, headers=headers, tablefmt="grid"))