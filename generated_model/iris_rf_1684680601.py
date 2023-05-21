
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss
import optuna

warnings.filterwarnings('ignore')

data = pd.read_csv('C:/Users/na018/Desktop/capstone/2023-1-Capstone-Design/iris.csv')
data.drop(['Id'], axis=1, inplace=True)

categorical_cols = []
for col in data.columns:
    if data[col].dtype == 'object':
        categorical_cols.append(col)

if len(categorical_cols) > 0:
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

X = data.drop(['Species'], axis=1)
y = data['Species']
le_y = LabelEncoder().fit_transform(y)

scaler_flag = True
if scaler_flag:
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

def objective(trial):
    max_depth = trial.suggest_int("max_depth", 10, 100)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 150)
    n_estimators = trial.suggest_int("n_estimators", 100, 2500)
    criterion = trial.suggest_categorical("criterion", ["gini","entropy"])
    random_state = 42
    
    model = RandomForestClassifier(max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   n_estimators=n_estimators,
                                   criterion=criterion,
                                   random_state=random_state)
    X_train, X_valid, y_train, y_valid = train_test_split(X, le_y, test_size=0.1, stratify=le_y, random_state=random_state)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    y_pred_proba = model.predict_proba(X_valid)
    score = f1_score(y_valid, y_pred, average='macro')
    return log_loss(y_valid, y_pred_proba), score

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=100)

print(study.best_params)
