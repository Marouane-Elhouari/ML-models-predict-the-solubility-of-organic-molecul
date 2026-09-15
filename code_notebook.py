from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import seaborn as snsw
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings("ignore")

import seaborn as sns
df = pd.read_csv(r"C:\Users\pc\Desktop\project\tutorial_rdkit\delaney_mordred_truncated.csv")

y = df['measured log(solubility:mol/L)']
X = df.drop(['measured log(solubility:mol/L)', 'Compound ID', 'SMILES'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
print(f'The r2 score for train set is : {rf.score(X_train, y_train)}')
print(f'The r2 score for test set is : {rf.score(X_test, y_test)}')
# get importance
importance = rf.feature_importances_
# summarize feature importance
dicts = {
    'Features':[x for x in df_final.iloc[:,3:].columns],
    'Importance':importance
    }
DF_imp = pd.DataFrame(dicts)
DF_imp = DF_imp.sort_values('Importance',ascending=False)
DF_imp.to_excel('imp.xlsx', index=None)

# plot feature importance
top_desc_fi = DF_imp[:8]
plt.subplots(figsize=(6,6))
sns.barplot(data=top_desc_fi, x = 'Features', y='Importance', palette = 'Set2')

plt.xticks(rotation = 90)
plt.show()
print(top_desc_fi.values)


df = df[top_desc_fi['Features'][:8]]

df_new = pd.concat([df , y]  ,  axis  =1)

df_new.to_csv( 'delaney_8_des.csv', index = None)



df_final = pd.read_csv(r'C:\Users\pc\Desktop\project\tutorial_rdkit\models_prediction_solubulity\delaney_8_des.csv')

y = df_final['measured log(solubility:mol/L)']
X = df_final.drop(['measured log(solubility:mol/L)', 'Compound ID', 'SMILES'], axis=1)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
print(f'The r2 score for train set is : {rf.score(X_train, y_train)}')
print(f'The r2 score for test set is : {rf.score(X_test, y_test)}')

et = ExtraTreesRegressor()
et.fit(X_train, y_train)
print(f'The r2 score for train set is : {et.score(X_train, y_train)}')
print(f'The r2 score for test set is : {et.score(X_test, y_test)}')
from sklearn.model_selection import GridSearchCV  , RandomizedSearchCV
rf = RandomForestRegressor(n_estimators=100,
                           bootstrap=True,# used the draw and sampling with replacement
                           max_depth=None,
                           max_features=1.0,
                           min_samples_leaf=1,
                           min_samples_split=2,
                          )
clf_rf = GridSearchCV(estimator=rf, param_grid = { 'n_estimators': [400, 600, 700, 800, 900],
                                                  'bootstrap': [True, False],
                                                  'max_depth': [30, 45, 60, 75, 100, None],
                                                  'max_features': ['log2', 'sqrt'],
                                                  'min_samples_leaf': [2, 4],
                                                  'min_samples_split': [5, 10],
                                                  }, cv=3, verbose=0, scoring='r2')
clf_rf.fit(X_train, y_train)
clf_rf.cv_results_
clf_rf_df = pd.DataFrame(clf_rf.cv_results_).sort_values('mean_test_score' , ascending= False).head()


print(f'The r2 score for train set is : {clf_rf.score(X_train, y_train)}')
print(f'The r2 score for test set is : {clf_rf.score(X_test, y_test)}')
print(f"The best parameters are: {clf_rf.best_params_}")

et = ExtraTreesRegressor(
                          n_estimators=100,
                           bootstrap=True,# used the draw and sampling with replacement
                           max_depth=None,
                           max_features=1.0,
                           min_samples_leaf=1,
                           min_samples_split=2,
                          )



clr_et = GridSearchCV(estimator=et, param_grid = { 'n_estimators': [400, 600, 700, 800, 900],
                                                  'bootstrap': [True, False],
                                                  'max_depth': [30, 45, 60, 75, 100, None],
                                                  'max_features': ['log2', 'sqrt'],
                                                  'min_samples_leaf': [2, 4],
                                                  'min_samples_split': [5, 10],
                                                  }, cv=3, verbose=0, scoring='r2')
clr_et.fit(X_train, y_train)
print(f'The r2 score for train set is : {clr_et.score(X_train, y_train)}')
print(f'The r2 score for test set is : {clr_et.score(X_test, y_test)}')
print(f"The best parameters are: {clf_rf.best_params_}")