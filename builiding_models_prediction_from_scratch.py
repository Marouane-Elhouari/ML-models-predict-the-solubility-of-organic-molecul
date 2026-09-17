import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools, AllChem

import mordred
from mordred import Calculator, descriptors
import pandas as pd
import numpy as np
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

SEED = 42
TARGET_COL = "measured log(solubility:mol/L)"
N_SPLITS = 4
N_JOBS = -1
N_TRIALS = 30


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed()

yellow, cyan_g, cyan_dark = "#F7C53E", "#0CF7AF", "#11AB7C"
purple, purple_dark, purple_light = "#D826F8", "#9309AB", "#b683d6"
blue, red, orange, green = "#0C97FA", "#FA1D19", "#FA9F19", "#0CFA58"
light_blue, soft_blue, dark_blue = "#01FADC", "#81c9e6", "#394be6"

PALETTE_2 = [cyan_g, purple]
PALETTE_3 = [yellow, cyan_g, purple]
PALETTE_7 = [purple_dark, purple_light, purple, blue, light_blue, dark_blue, soft_blue]

sns.set_style("whitegrid")
sns.set_palette(PALETTE_7)
plt.rcParams["figure.facecolor"] = "#f8fafc"
pd.set_option("display.float_format", "{:.4f}".format)


DATA_PATH = r'C:\Users\pc\Desktop\project\tutorial_rdkit\delaney_mordred_truncated.csv'
df_3d_mol = pd.read_csv(DATA_PATH)
df_3d_mol.head()


def build_eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = df[col]
        not_null = series.dropna()
        rows.append({
            'features': col,
            'dtype': str(series.dtype),
            'missing_count': int(series.isna().sum()),
            'missing_pct': float(series.isna().mean() * 100.0),
            'nunique': int(series.nunique()),
            'sample_values': ','.join(not_null.astype(str).unique()[:4]),
        })
    return pd.DataFrame(rows)

eda_cols = [c for c in df_3d_mol.columns if c != TARGET_COL]
eda_summary = build_eda_summary(df_3d_mol[eda_cols])
eda_summary


df_3d_mol.isna().sum()


print(df_3d_mol.duplicated().sum())
df_3d_mol.drop_duplicates(inplace=True)
print(df_3d_mol.duplicated().sum())


column_num = []
column_bool = []
for column in df_3d_mol.columns:
    if column == TARGET_COL:
        continue
    column_type = df_3d_mol[column].dtype
    if column_type == 'object' or column_type == 'string':
        pass
    elif column_type == 'bool':
        column_bool.append(column)
    else:
        column_num.append(column)

data = df_3d_mol[column_num + column_bool]
data.shape


def remove_constant_values(data):
    return [e for e in data.columns if data[e].nunique() == 1]

drop_col = remove_constant_values(data)
new_df_columns = [e for e in data.columns if e not in drop_col]
new_df = data[new_df_columns].copy()
new_df.shape


def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

corr_features = correlation(new_df, 0.9)
new_df.drop(columns=corr_features, inplace=True)
new_df.shape


def convert_bool_to_num(df: pd.DataFrame, bool_cols: list) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        if column in bool_cols:
            df[column] = df[column].astype(int)
    return df

df_new = convert_bool_to_num(new_df, column_bool)
df_new[TARGET_COL] = df_3d_mol.loc[df_new.index, TARGET_COL]
df_new.to_csv('new_df.csv', index=None)
df_new.head()


EDA_SAMPLE = 5000
NUMERICAL_COL_RAW = [
    'ABC', 'nBase', 'SpMAD_A', 'VR1_A', 'nAromAtom', 'nSpiro', 'nBridgehead', 'nHetero', 'nN', 'nO', 'nS', 'nP', 'nF', 'nCl', 'nBr', 'nI', 'ATS0Z', 'AATS0dv', 'AATS0Z', 'AATS0se', 'AATS0i', 'ATSC2c', 'ATSC4c', 'ATSC5c', 'ATSC6c', 'ATSC7c', 'ATSC8c', 'ATSC1dv', 'ATSC2dv', 'ATSC3dv', 'ATSC4dv', 'ATSC5dv', 'ATSC6dv', 'ATSC7dv', 'ATSC8dv', 'ATSC2d', 'ATSC3d', 'ATSC4d', 'ATSC5d', 'ATSC6d', 'ATSC7d', 'ATSC8d', 'ATSC1Z', 'ATSC2Z', 'ATSC3Z', 'ATSC4Z', 'ATSC5Z', 'ATSC6Z', 'ATSC7Z', 'ATSC8Z', 'ATSC1v', 'ATSC2v', 'ATSC3v', 'ATSC4v', 'ATSC5v', 'ATSC6v', 'ATSC7v', 'ATSC8v', 'ATSC1se', 'ATSC3se', 'ATSC4se', 'ATSC5se', 'ATSC6se', 'ATSC7se', 'ATSC8se', 'ATSC1p', 'ATSC1i', 'ATSC2i', 'ATSC3i', 'ATSC5i', 'ATSC6i', 'ATSC7i', 'ATSC8i', 'AATSC1d', 'AATSC2Z', 'AATSC0v', 'AATSC0i', 'MATS1c', 'MATS1Z', 'MATS2Z', 'GATS2c', 'GATS1Z', 'GATS2Z', 'GATS2v', 'GATS1se', 'GATS2se', 'GATS2i', 'BCUTc-1l', 'BCUTdv-1l', 'BCUTd-1h', 'BCUTd-1l', 'BCUTZ-1l', 'BalabanJ', 'nBondsD', 'nBondsT', 'FPSA3', 'RNCG', 'RPCG', 'RNCS', 'RPCS', 'C2SP1', 'C1SP2', 'C3SP2', 'C1SP3', 'C2SP3', 'C3SP3', 'C4SP3', 'FCSP3', 'Xch-3d', 'Xch-4d', 'Xc-4d', 'Xc-4dv', 'NsCH3', 'NdCH2', 'NtCH', 'NdsCH', 'NsssCH', 'NaasC', 'NaaaC', 'NsNH2', 'NdNH', 'NssNH', 'NaaNH', 'NtN', 'NdsN', 'NaaN', 'NsssN', 'NddsN', 'NaasN', 'NsOH', 'NssO', 'NaaO', 'NsSH', 'NdS', 'NssS', 'NaaS', 'NdssS', 'NddssS', 'SsssCH', 'SdssC', 'SaasC', 'ETA_shape_y', 'ETA_beta_ns_d', 'AETA_eta', 'AETA_eta_RL', 'ETA_dEpsilon_D', 'ETA_dBeta', 'GeomShapeIndex', 'nHBDon', 'IC0', 'IC1', 'SIC0', 'SIC1', 'Mor02', 'Mor06', 'Mor07', 'Mor08', 'Mor09', 'Mor10', 'Mor11', 'Mor13', 'Mor15', 'Mor16', 'Mor20', 'Mor21', 'Mor22', 'Mor23', 'Mor24', 'Mor25', 'Mor26', 'Mor27', 'Mor28', 'Mor30', 'Mor32', 'Mor02m', 'Mor03m', 'Mor04m', 'Mor05m', 'Mor06m', 'Mor09m', 'Mor10m', 'Mor11m', 'Mor12m', 'Mor13m', 'Mor14m', 'Mor15m', 'Mor16m', 'Mor17m', 'Mor18m', 'Mor19m', 'Mor20m', 'Mor21m', 'Mor22m', 'Mor23m', 'Mor24m', 'Mor25m', 'Mor26m', 'Mor27m', 'Mor28m', 'Mor29m', 'Mor30m', 'Mor31m', 'Mor32m', 'Mor04v', 'Mor10v', 'Mor11v', 'Mor16v', 'Mor22v', 'Mor24v', 'Mor30v', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'SMR_VSA6', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA7', 'SlogP_VSA10', 'SlogP_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'VSA_EState1', 'VSA_EState4', 'VSA_EState5', 'VSA_EState9', 'AMID_O', 'PBF', 'n5Ring', 'n7Ring', 'n8Ring', 'nHRing', 'n3HRing', 'n5HRing', 'n5aRing', 'nAHRing', 'n5AHRing', 'n6AHRing', 'nFRing', 'n6FRing', 'n7FRing', 'n8FRing', 'n9FRing', 'n10FRing', 'n11FRing', 'n12FRing', 'nG12FRing', 'nFHRing', 'n10FHRing', 'n12FHRing', 'nG12FHRing', 'n9FaRing', 'nFaHRing', 'nG12FaHRing', 'nFARing', 'n9FARing', 'n10FARing', 'nFAHRing', 'n10FAHRing', 'nRot', 'JGI1', 'JGI2', 'JGI3', 'JGI4', 'JGI5', 'JGI6', 'JGI7', 'JGI8', 'JGI9', 'JGI10', 'Lipinski', 'GhoseFilter'
]
NUMERICAL_COL = [c for c in NUMERICAL_COL_RAW if c in df_new.columns]

eda_sample = df_new.sample(min(len(df_new), EDA_SAMPLE), random_state=SEED)

n_cols = 2
n_rows = int(np.ceil(len(NUMERICAL_COL) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.2 * n_rows), constrained_layout=True)
axes = axes.flatten()

for i, col in enumerate(NUMERICAL_COL):
    sns.regplot(
        data=eda_sample, x=col, y=TARGET_COL,
        scatter_kws={'alpha': 0.35, 's': 15, 'color': "#0CF7AF"},
        line_kws={'color': "#D826F8"},
        ax=axes[i]
    )
    corr_val = eda_sample[[col, TARGET_COL]].corr().iloc[0, 1]
    axes[i].set_title(f'{col} vs {TARGET_COL} (corr={corr_val:.2f})', fontsize=10)

for j in range(len(NUMERICAL_COL), len(axes)):
    axes[j].axis('off')
plt.show()


corr = df_new.corr()
corr_sorted = abs(corr[[TARGET_COL]]).sort_values(by=TARGET_COL, ascending=False)
corr_sorted.rename(columns={TARGET_COL: 'correlation_coef'}, inplace=True)
corr_sorted.iloc[1:5, :]


y = df_new[TARGET_COL]
X = df_new.drop(columns=[TARGET_COL])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

print(X_train.shape)
print(X_test.shape)


kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

def evaluate_all_metrics(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred, squared=False),
        'R2': r2_score(y_true, y_pred),
    }


def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
        'random_state': SEED,
        'n_jobs': N_JOBS,
    }
    scores = []
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        scores.append(mean_absolute_error(y_val, preds))
    return np.mean(scores)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(xgb_objective, n_trials=N_TRIALS)
print('Best XGB MAE (CV):', study_xgb.best_value, '| params:', study_xgb.best_params)


def lgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
        'random_state': SEED,
        'n_jobs': N_JOBS,
        'verbose': -1,
    }
    scores = []
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        scores.append(mean_absolute_error(y_val, preds))
    return np.mean(scores)

study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(lgb_objective, n_trials=N_TRIALS)
print('Best LGBM MAE (CV):', study_lgb.best_value, '| params:', study_lgb.best_params)


def cat_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 500, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1, log=True),
        'depth': trial.suggest_int('depth', 3, 6),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_state': SEED,
        'silent': True,
    }
    scores = []
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        scores.append(mean_absolute_error(y_val, preds))
    return np.mean(scores)

study_cat = optuna.create_study(direction='minimize')
study_cat.optimize(cat_objective, n_trials=N_TRIALS)
print('Best CatBoost MAE (CV):', study_cat.best_value, '| params:', study_cat.best_params)


models = {
    'XGBoost': XGBRegressor(**study_xgb.best_params, random_state=SEED, n_jobs=N_JOBS, tree_method='hist'),
    'LightGBM': LGBMRegressor(**study_lgb.best_params, random_state=SEED, verbose=-1, n_jobs=N_JOBS),
    'CatBoost': CatBoostRegressor(**study_cat.best_params, random_state=SEED, silent=True, thread_count=N_JOBS),
}

all_importances = {}

for name, model in models.items():
    print(f"Training and evaluating: {name}...")
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    scores = evaluate_all_metrics(y_test, y_preds)
    print(f"--- {name} ---")
    for metric_name, value in scores.items():
        print(f"{metric_name}: {value:.4f}")

    fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True)
    ax.scatter(y_test, y_preds, alpha=0.4, color=cyan_g, s=20)
    lims = [min(y_test.min(), y_preds.min()), max(y_test.max(), y_preds.max())]
    ax.plot(lims, lims, color=purple, linestyle='--')
    ax.set_xlabel('Valeur reelle')
    ax.set_ylabel('Valeur predite')
    ax.set_title(f'Prédit vs Réel — {name}', fontsize=11)
    plt.show()

    importance = model.feature_importances_
    df_imp = pd.DataFrame({
        'Features': X_train.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    df_imp.to_excel(f'imp_{name}.xlsx', index=None)
    all_importances[name] = df_imp

    top_8 = df_imp.head(8)
    plt.figure(figsize=(7, 5))
    sns.barplot(data=top_8, x='Features', y='Importance', hue='Features', legend=False, palette='Set2')
    plt.title(f'Top 8 Feature Importances - {name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.close()


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
import pandas as pd
import numpy as np
scaler = StandardScaler()
def predict_sol_smiles(smiles):

    if isinstance(smiles, str):
        smiles = [smiles]

    valid_smiles = []
    mols = []
    for smile in smiles:
        mol = _build_mol(smile)
        if mol is not None:
            valid_smiles.append(smile)
            mols.append(mol)

    if not mols:
        print("[ERROR] No valid molecules to predict on.")
        return pd.DataFrame(columns=['SMILES'])

    calc = Calculator(descriptors, ignore_3D=False)
    desc = calc.pandas(mols)

    desc_all = desc.apply(pd.to_numeric, errors='coerce')


    if desc_all.isna().any().any():
        n_bad = int(desc_all.isna().sum().sum())
        print(f"[WARN] {n_bad} descriptor value(s) could not be computed, filled with column mean.")
        desc_all = desc_all.fillna(desc_all.mean()).fillna(0)

    X_new = scaler.transform(desc_all)


    results_df = pd.DataFrame({'SMILES': valid_smiles})

    for name, model in models.items():
        predicted_logS = model.predict(X_new)
        results_df[f'logS_{name}'] = predicted_logS
        results_df[f'Solubility_mol_L_{name}'] = 10 ** predicted_logS

    sol_cols = [c for c in results_df.columns if c.startswith('Solubility_mol_L_')]
    results_df['Ensemble_Solubility_mol_L'] = results_df[sol_cols].mean(axis=1)

    return results_df