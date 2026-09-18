

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors



def _build_mol_3d(smile):

    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print("[WARN] SMILES invalide, ignore : " + smile)
        return None

    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, randomSeed=42)
    if status != 0:
        status = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    if status != 0:
        print("[WARN] Embedding 3D echoue, ignore : " + smile)
        return None
    return mol


def predict_solubility(smiles_input, models, feature_columns, train_means, verbose=True):

    if isinstance(smiles_input, str):
        smiles_input = [smiles_input]

    valid_smiles, mols = [], []
    for smi in smiles_input:
        mol = _build_mol_3d(smi)
        if mol is not None:
            valid_smiles.append(smi)
            mols.append(mol)

    if not mols:
        print("[ERROR] Aucune molecule valide a predire.")
        return pd.DataFrame(columns=["SMILES"])

    calc = Calculator(descriptors, ignore_3D=False)
    raw = calc.pandas(mols)
    desc_num = raw.apply(pd.to_numeric, errors="coerce")

    if verbose:
        print("[INFO] " + str(desc_num.shape[1]) + " descripteurs Mordred calcules")


    X = desc_num.reindex(columns=feature_columns)

    n_nan = int(X.isna().sum().sum())
    if n_nan > 0:
        if verbose:
            bad_cols = X.columns[X.isna().any()].tolist()
            print("[WARN] " + str(n_nan) + " valeur(s) NaN -> remplacement par moyenne d'entrainement")
            print("       Descripteurs concernes : " + str(bad_cols))
        X = X.fillna(train_means).fillna(0)

    X = pd.DataFrame(
        np.asarray(X, dtype=float),
        columns=feature_columns,
        index=X.index
    )


    results = pd.DataFrame({"SMILES": valid_smiles})

    for name, model in models.items():
        log_s = model.predict(X)
        results["logS_" + name] = log_s
        results["Solubility_mol_L_" + name] = 10.0 ** log_s

 
    sol_cols  = [c for c in results.columns if c.startswith("Solubility_mol_L_")]
    logS_cols = [c for c in results.columns if c.startswith("logS_")]

    results["Ensemble_Solubility_mol_L"] = results[sol_cols].mean(axis=1)
    results["Ensemble_logS"]             = results[logS_cols].mean(axis=1)

    return results



def check_artifacts(feature_columns, train_means, models, verbose=True):

    ok = True

    if hasattr(train_means, "index"):
        missing_means = [f for f in feature_columns if f not in train_means.index]
    else:
        missing_means = [f for f in feature_columns if f not in train_means]

    if missing_means:
        print("[ERROR] " + str(len(missing_means)) + " features absentes de train_means : " + str(missing_means[:5]))
        ok = False
    elif verbose:
        print("[OK] train_means couvre les " + str(len(feature_columns)) + " features")

    n_expected = len(feature_columns)

    for name, model in models.items():
        try:
            n_feat = model.n_features_in_

            if n_feat == 0:
                if verbose:
                    print("[INFO] " + name + " : n_features_in_=0 (CatBoost), verification ignoree")
            elif n_feat != n_expected:
                print("[ERROR] " + name + " attend " + str(n_feat) +
                      " features, mais feature_columns en a " + str(n_expected))
                ok = False
            else:
                if verbose:
                    print("[OK] " + name + " : " + str(n_feat) + " features OK")

        except AttributeError:
            if verbose:
                print("[INFO] " + name + " : attribut n_features_in_ absent, verification ignoree")

    return ok
