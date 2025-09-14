import json
# import pandas as pd
import numpy as np
from sklearn import datasets
from evidently.report import Report
from evidently.metrics import DataDriftTable
from evidently.metrics import DatasetDriftMetric

# --- (1) Charger le jeu de données
adult_data = datasets.fetch_openml(name="adult", version=2, as_frame="auto")
adult = adult_data.frame

# (Optionnel) Renommer explicitement les colonnes si tu veux des libellés propres dans le rapport
# ⚠️ Fais exactement le même rename sur reference et current pour garder les schémas identiques
rename_map = {
    "education-num": "education_num",
    "hours-per-week": "hours_per_week",
    "capital-gain": "capital_gain",
    "capital-loss": "capital_loss",
    "native-country": "native_country",
    "marital-status": "marital_status",
}
adult = adult.rename(columns=rename_map)

# --- (2) Split reference / current
adult_ref = adult[~adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]
adult_cur = adult[adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]

# Introduire des valeurs manquantes pour la démonstration
adult_cur.iloc[:2000, 3:5] = np.nan

# --- (3) Définir un ColumnMapping explicite (rôles + types)
# Import compatible suivant ta version d'Evidently
try:
    from evidently import ColumnMapping
except ImportError:
    from evidently.pipeline.column_mapping import ColumnMapping  # fallback versions plus anciennes

column_mapping = ColumnMapping(
    target="class",                  # colonne cible (revenu >50K ?)
    prediction=None,                 # si tu as des prédictions, mets le nom ici
    id=None,
    datetime=None,
    # Liste explicite des numériques :
    numerical_features=[
        "age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"
    ],
    # Liste explicite des catégorielles :
    categorical_features=[
        "workclass", "education", "marital_status", "occupation",
        "relationship", "race", "sex", "native_country"
    ],
    # (Optionnel) Spécifier des ordinales avec un ordre :
    # ordinal_features={"education": ["Preschool","1st-4th","5th-6th","7th-8th","9th","10th",
    #                                 "11th","12th","HS-grad","Some-college","Assoc-voc","Assoc-acdm",
    #                                 "Bachelors","Masters","Prof-school","Doctorate"]},
    text_features=[],
)

# --- (4) Construire et exécuter le rapport en passant column_mapping
data_drift_dataset_report = Report(metrics=[
    DatasetDriftMetric(),
    DataDriftTable(),
])

data_drift_dataset_report.run(
    reference_data=adult_ref,
    current_data=adult_cur,
    column_mapping=column_mapping,   # ⟵ le mapping explicite est utilisé ici
)

# --- (5) Sauvegardes
report_data = json.loads(data_drift_dataset_report.json())
with open("data_drift_report.json", "w") as f:
    json.dump(report_data, f, indent=4)

data_drift_dataset_report.save_html("Data drift report.html")
# En notebook :
# data_drift_dataset_report.show()
