import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import optuna
from xgboost import XGBClassifier

data = pd.read_csv('C:/Users/na018/Desktop/capstone/2023-1-Capstone-Design/iris.csv')
data.drop(['Id'], axis=1, inplace=True)

categorical_cols = []
if set(categorical_cols).issubset(set(data.columns)):
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

label = 'Species'
le = LabelEncoder()
data[label] = le.fit_transform(data[label])

scaler_flag = True
if scaler_flag:
    scaler = MinMaxScaler()
    data[data.columns] = scaler.fit_transform(data[data.columns])

def objective(trial):
    params = {
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-5, 1.0),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-5, 1.0),
        'max_depth': trial.suggest_int("max_depth", 4,100),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1),
        'subsample': trial.suggest_loguniform("subsample", 0.5, 1),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
        'n_estimators': trial.suggest_int("n_estimators", 100, 3000),
        'min_child_samples': trial.suggest_int("min_child_samples", 5, 100),
        'random_state': 42,
        'num_class': 3
    }
    
    X_train, X_valid, y_train, y_valid = train_test_split(data.drop(labels=[label], axis=1), data[label], test_size=0.1, random_state=42, stratify=data[label])

    model = XGBClassifier(**params, objective='multi:softprob')
    model.fit(X_train, y_train, verbose=False, eval_set=[(X_valid, y_valid)], eval_metric='mlogloss', early_stopping_rounds=100)

    preds_valid = model.predict(X_valid)
    f1 = f1_score(y_valid, preds_valid, average='macro')

    return f1

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=100)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))