
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss
import optuna

#load data
data = pd.read_csv('C:/Users/na018/Desktop/capstone/2023-1-Capstone-Design/iris.csv')
y = data['Species']
X = data.drop(['Id', 'Species'], axis = 1)

#encode categorical columns
le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col])

#encode label data
y = le.fit_transform(y)

#scale data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

#optimize hyperparameters
def objective(trial):
    params = {
        'max_depth': trial.suggest_int("max_depth", 10, 100),
        'min_samples_split': trial.suggest_int("min_samples_split", 2, 150),
        'n_estimators': trial.suggest_int("n_estimators", 100, 2500),
        'criterion': trial.suggest_categorical("criterion", ["gini","entropy"]),
        'bootstrap': True,
        'random_state': 42
    }
    model = RandomForestClassifier(**params)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, stratify = y, random_state = 42)
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    return f1_score(y_val, y_pred_val, average='macro')

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(sampler = sampler, direction='maximize')
study.optimize(objective, n_trials = 50)
