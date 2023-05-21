import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from warnings import simplefilter
import optuna

simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv('C:/Users/na018/Desktop/capstone/2023-1-Capstone-Design/iris.csv')

data.drop(['Id'], axis=1, inplace=True)

categorical_columns = [col for col in data.columns if data[col].dtype == 'object']

if categorical_columns:
    le = LabelEncoder()
    for col in categorical_columns:
        data[col] = le.fit_transform(data[col])

X = data.drop('Species', axis=1)
y = data['Species']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

def objective(trial):
    params = {
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-5, 1.0),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-5, 1.0),
        'max_depth': trial.suggest_int("max_depth", 4,100),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1),
        'subsample': trial.suggest_float("subsample", 0.5, 1),
        'learning_rate': trial.suggest_float("learning_rate",1e-5, 1e-1),
        'n_estimators': trial.suggest_int("n_estimators", 100, 3000),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 50),
        'gpu_id': -1,
        'random_state': 42,
        'num_class': 3
    }
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.1, random_state=42, stratify=y)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=100)
    score = f1_score(y_valid, model.predict(X_valid), average='macro')
    return score

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

study.optimize(objective, n_trials=100)

params = study.best_params
params['gpu_id'] = -1
params['random_state'] = 42
params['num_class'] = 3

model = XGBClassifier(**params)
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.1, random_state=42, stratify=y)

model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='logloss', early_stopping_rounds=100)

final_score = f1_score(y_valid, model.predict(X_valid), average='macro')

print(final_score)