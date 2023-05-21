import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import optuna
from optuna.samplers import TPESampler

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

SEED = 42

data = pd.read_csv("C:/Users/na018/Desktop/capstone/2023-1-Capstone-Design/iris.csv")
data.drop('Id', axis=1, inplace=True)

categorical_cols = []
for col in data.columns:
    if data[col].dtype == 'object':
        categorical_cols.append(col)

if categorical_cols:
    labelencoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = labelencoder.fit_transform(data[col])

X = data.drop('Species', axis=1)
y = data['Species']

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

def objective(trial):
    params = {'reg_lambda': trial.suggest_float("reg_lambda", 1e-5, 1.0),
              'reg_alpha': trial.suggest_float("reg_alpha", 1e-5, 1.0),
              'max_depth': trial.suggest_int("max_depth", 4,100),
              'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1),
              'subsample': trial.suggest_loguniform("subsample", 0.5, 1),
              'learning_rate': trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
              'n_estimators': trial.suggest_int("n_estimators", 100, 3000),
              'min_child_samples': trial.suggest_int("min_child_samples", 5, 100),
              'random_state': 42, 'num_class': 3}
    
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)
    
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)
    
    model = XGBClassifier(**params)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric='mlogloss', verbose=0, early_stopping_rounds=100)
    
    predictions = model.predict(x_valid)
    
    score = f1_score(y_valid, predictions, average='macro')
    
    return score

sampler = TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=100)

best_params = study.best_params
print(best_params)