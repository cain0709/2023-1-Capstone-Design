
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, f1_score
import optuna
from optuna.samplers import TPESampler

SEED = 42

#load data
data = pd.read_csv("C:/Users/na018/Desktop/capstone/2023-1-Capstone-Design/iris.csv")

#drop useless features
data.drop(['Id'], axis=1, inplace=True)

#scale data
scaler_flag = True
if scaler_flag:
    scaler = MinMaxScaler()
    data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = scaler.fit_transform(data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])

#if categorical variables exist, use sklearn labelencoder
categorical_cols = []
if len(categorical_cols) > 0:
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

#encode label data y with labelencoder
le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])

#split data into X and y
X = data.drop(['Species'], axis=1)
y = data['Species']

#split data into train and validation data using stratified sampling
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)

#define the objective function of optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2500),
        "max_depth": trial.suggest_int("max_depth", 10, 100),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 150),
        "criterion": trial.suggest_categorical("criterion", ["gini","entropy"]),
        "random_state": SEED
        }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    f1 = f1_score(y_valid, y_pred, average='macro')
    logloss = log_loss(y_valid, model.predict_proba(X_valid))
    return f1

#optimize hyperparameters using optuna
sampler = TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=100)

#print best hyperparameters and score, valid f1 score and valid mse
print("Best hyperparameters: {}".format(study.best_params))
print("Best score: {}".format(study.best_value))
best_params = study.best_params
model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
f1 = f1_score(y_valid, y_pred, average='macro')
logloss = log_loss(y_valid, model.predict_proba(X_valid))
print("Valid F1-score: {}".format(f1))
print("Valid Log-loss: {}".format(logloss))
