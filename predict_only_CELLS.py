
import os
import sys
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from mordred import Calculator, descriptors

ARTIFACTS_DIR = r"C:\Users\pc\Desktop\project\tutorial_rdkit\models_prediction_solubulity"
sys.path.insert(0, ARTIFACTS_DIR)

# Importer les fonctions du module de prediction
from prediction_solubility_from_smiles import predict_solubility, check_artifacts

feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))
train_means     = joblib.load(os.path.join(ARTIFACTS_DIR, "train_means.pkl"))
models          = joblib.load(os.path.join(ARTIFACTS_DIR, "models.pkl"))

print("Modeles charges       :", list(models.keys()))
print("Nombre de features    :", len(feature_columns))
print()

artifacts_ok = check_artifacts(feature_columns, train_means, models, verbose=True)
if not artifacts_ok:
    raise RuntimeError("Artefacts incoherents - verifie les .pkl avant de continuer.")


validation_smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",          # Aspirine
    "CCO",                              # Ethanol
    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",     # Cafeine
    "c1ccc2ccccc2c1",                  # Naphtalene
]
validation_names  = ["Aspirine", "Ethanol", "Cafeine", "Naphtalene"]
logS_experimentaux = [-2.2, 0.0, -1.0, -3.6]

print("\n--- Validation sur molecules de reference ---")
results_val = predict_solubility(
    validation_smiles, models, feature_columns, train_means, verbose=True
)
results_val.insert(1, "Nom", validation_names)
results_val["logS_exp"]          = logS_experimentaux
results_val["Erreur_vs_exp"]     = (results_val["Ensemble_logS"] - results_val["logS_exp"]).round(2)

cols_show = ["Nom", "Ensemble_logS", "logS_exp", "Erreur_vs_exp", "Ensemble_Solubility_mol_L"]
print(results_val[cols_show].to_string(index=False))



my_smiles = [
    "OCC1=C(O)C=C(O)C=C1",    
    "NCC1=C(N)C=C(N)C=C1",    
    "CCC1=C(C)C=C(C)C=C1",    
    
]

print("\n--- Predictions ---")
df = predict_solubility(my_smiles, models, feature_columns, train_means, verbose=True)
print(df.to_string(index=False))
