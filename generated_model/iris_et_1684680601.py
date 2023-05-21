import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import optuna
import warnings

warnings.filterwarnings('ignore')

# Loading data
data = pd.read_csv('C:/Users/na018/Desktop/capstone/2023-1-Capstone-Design/iris.csv')
# Dropping useless columns
data.drop(['Id'], axis=1, inplace=True)

# Encoding label data
le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])

# Scaling data
scaler = MinMaxScaler()
data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])

# Objective function for optuna
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 10, 100),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2500),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'bootstrap': True,
        'random_state': 42
    }
    model = RandomForestClassifier(**params)
    X_train, X_val, y_train, y_val = train_test_split(data[data.columns[:-1]], data['Species'], test_size=0.1, stratify=data['Species'], random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    return f1

# Optimizing hyperparameters
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)

# Best hyperparameters
best_params = study.best_params

# Final model
final_model = RandomForestClassifier(**best_params)
X_train, X_val, y_train, y_val = train_test_split(data[data.columns[:-1]], data['Species'], test_size=0.1, stratify=data['Species'], random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_val)

# Final evaluation metric
from sklearn.metrics import f1_score, log_loss
print('Macro F1-score:', f1_score(y_val, y_pred, average='macro'))
print('Log loss:', log_loss(y_val, final_model.predict_proba(X_val)))