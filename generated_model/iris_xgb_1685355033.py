import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, log_loss
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')

data = pd.read_csv('C:/Users/na018/Desktop/capstone/2023-1-Capstone-Design/iris.csv')
data.drop(['Id'], axis=1, inplace=True)

scaler_flag = True

if scaler_flag:
    scaler = MinMaxScaler()
    data[data.columns] = scaler.fit_transform(data[data.columns])

categorical_columns = []
for col, col_type in data.dtypes.iteritems():
    if col_type == 'O':
        categorical_columns.append(col)
        
if categorical_columns:
    for col in categorical_columns:
        data[col] = LabelEncoder().fit_transform(data[col])
    
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

def objective(trial):
    params = {
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0),
        "max_depth": trial.suggest_int("max_depth", 4, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1),
        "n_estimators": trial.suggest_int("n_estimators", 100, 3000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "gpu_id": -1,
        "random_state": 42,
        "num_class": 3
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], 
              early_stopping_rounds=50, eval_metric=["mlogloss", "merror"], verbose=False)
    
    y_pred = model.predict(X_valid)
    f1 = f1_score(y_valid, y_pred, average='macro')
    logloss = log_loss(y_valid, model.predict_proba(X_valid))
    
    return f1

sampler = TPESampler(seed=42)

study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=100)

print('Best hyperparameters:', study.best_params)
print('Best score:', study.best_value)
print('Valid f1 score:', f1_score(y_valid, model.predict(X_valid), average='macro'))
print('Valid mse:', log_loss(y_valid, model.predict_proba(X_valid)))