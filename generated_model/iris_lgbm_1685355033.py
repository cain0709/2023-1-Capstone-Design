
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, f1_score
from warnings import filterwarnings
import optuna
from optuna.samplers import TPESampler
from lightgbm import LGBMClassifier

filterwarnings('ignore')

data = pd.read_csv('C:/Users/na018/Desktop/capstone/2023-1-Capstone-Design/iris.csv')
data.drop(['Id'], axis=1, inplace=True)

categorical_columns = []
numerical_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

scaler_flag = True
if categorical_columns:
    le = LabelEncoder()
    data[categorical_columns] = data[categorical_columns].apply(lambda col: le.fit_transform(col))
if scaler_flag:
    scaler = MinMaxScaler() 
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

y = data['Species']
X = data.drop(['Species'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_valid = le.transform(y_valid)

def objective(trial):
    params = {
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0),
        'max_depth': trial.suggest_int('max_depth', 4, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
        'subsample': trial.suggest_loguniform('subsample', 0.5, 1),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'random_state': 42,
        'num_class': 3
    }
    
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=100, verbose=False,
              eval_metric='multi_logloss')
    
    pred = model.predict(X_valid)
    f1 = f1_score(y_valid, pred, average='macro')
    
    return f1

sampler = TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=100)

print(study.best_params)
print(f"Best score: {study.best_value:.5f}")
print("Valid F1-score:", f1_score(y_valid, model.predict(X_valid), average='macro'))
print("Valid MSE:", log_loss(y_valid, model.predict_proba(X_valid)))