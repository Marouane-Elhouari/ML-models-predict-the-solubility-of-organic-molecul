import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

from mordred import Calculator, descriptors

DATA_PATH = r'C:\Users\pc\Desktop\project\tutorial_rdkit\models_prediction_solubulity\delaney_8_des.csv'

df = pd.read_csv(DATA_PATH)
print(df.head())

TARGET_COL = 'measured log(solubility:mol/L)'
y = df[TARGET_COL]
X = df.drop([TARGET_COL], axis=1)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(
    bootstrap=True,
    max_depth=45,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=5,
    n_estimators=800,
    random_state=0
)
rf.fit(X_train, y_train)

train_pred = rf.predict(X_train)
test_pred = rf.predict(X_test)

print(f'The r2 score for train set is : {r2_score(y_train, train_pred)}')
print(f'The r2 score for test set is : {r2_score(y_test, test_pred)}')

DESCRIPTOR_COLS = list(X.columns)

def _build_mol(smile: str):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print(f"[WARN] Invalid SMILES, skipped: {smile}")
        return None

    mol = Chem.AddHs(mol)
    embed_status = AllChem.EmbedMolecule(mol, randomSeed=42)
    if embed_status != 0:
        embed_status = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    if embed_status != 0:
        print(f"[WARN] 3D embedding failed, skipped: {smile}")
        return None

    return mol

def predict_sol_smiles(smiles):
    valid_smiles = []
    mols = []
    for smile in smiles:
        mol = _build_mol(smile)
        if mol is not None:
            valid_smiles.append(smile)
            mols.append(mol)

    if not mols:
        print("[ERROR] No valid molecules to predict on.")
        return pd.DataFrame(columns=['SMILES', 'Predicted_Solubility_mol_L'])

    df_mol = pd.DataFrame({'mol': mols})

    calc = Calculator(descriptors, ignore_3D=False)
    desc = calc.pandas(df_mol['mol'])

    desc_8 = desc[DESCRIPTOR_COLS]

    desc_8 = desc_8.apply(pd.to_numeric, errors='coerce')
    if desc_8.isna().any().any():
        n_bad = int(desc_8.isna().sum().sum())
        print(f"[WARN] {n_bad} descriptor value(s) could not be computed, filled with column mean.")
        desc_8 = desc_8.fillna(desc_8.mean())

    X_new = scaler.transform(desc_8)
    predicted_logS = rf.predict(X_new)
    predicted_solubility = 10 ** predicted_logS

    result = pd.DataFrame({
        'SMILES': valid_smiles,
        'Predicted_Solubility_mol_L': predicted_solubility,
    })
    return result
if __name__ == "__main__":
    smiles = ['OCC1=C(O)C=C(O)C=C1', 'NCC1=C(N)C=C(N)C=C1', 'CCC1=C(C)C=C(C)C=C1']
    mol_img = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        img = Draw.MolToImage(mol)
        mol_img.append(img)
    print('mol_img', mol_img)
    
    results_df = predict_sol_smiles(smiles)
    print(results_df)