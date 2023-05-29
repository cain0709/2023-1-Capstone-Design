
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, f1_score
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import ExtraTreesClassifier

SEED = 42

# Load data
data = pd.read_csv('C:/Users/na018/Desktop/capstone/2023-1-Capstone-Design/iris.csv')

# Drop useless features
data.drop(['Id'], axis=1, inplace=True)

# Encode categorical columns
labelencoder = LabelEncoder()
if len(data.select_dtypes(include=['object']).columns) > 0:
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = labelencoder.fit_transform(data[col])

# Split data into X and y
X = data.drop(['Species'], axis=1)
y = data['Species']

# Encode target variable
y = labelencoder.fit_transform(y)

# Scale data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)

# Define objective function for optuna
def objective(trial):
    
    # Set hyperparameters to be optimized
    max_depth = trial.suggest_int("max_depth", 10, 100)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 150)
    n_estimators = trial.suggest_int("n_estimators", 100, 2500)
    criterion = trial.suggest_categorical("criterion", ["gini","entropy"])
    bootstrap = True
    random_state = 42
    
    # Initialize ExtraTreesClassifier
    clf = ExtraTreesClassifier(max_depth=max_depth, min_samples_split=min_samples_split, n_estimators=n_estimators, criterion=criterion,
                               bootstrap=bootstrap, random_state=random_state)
    
    # Fit ExtraTreesClassifier to training data
    clf.fit(X_train, y_train)
    
    # Predict probabilities on validation set
    y_pred_valid_proba = clf.predict_proba(X_valid)
    
    # Calculate f1-score and log-loss using predicted probabilities
    f1 = f1_score(y_valid, y_pred_valid_proba.argmax(axis=1), average='macro')
    logloss = log_loss(y_valid, y_pred_valid_proba)
    
    return f1

# Set up optuna study
sampler = TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=100)

# Print best hyperparameters and score
best_params = study.best_params
best_score = study.best_value
print(f"Best hyperparameters: {best_params}")
print(f"Best score: {best_score}")

# Fit ExtraTreesClassifier with best hyperparameters
clf = ExtraTreesClassifier(max_depth=best_params["max_depth"], min_samples_split=best_params["min_samples_split"], 
                           n_estimators=best_params["n_estimators"], criterion=best_params["criterion"],
                           bootstrap=best_params["bootstrap"], random_state=best_params["random_state"])
clf.fit(X_train, y_train)

# Predict probabilities on validation set
y_pred_valid_proba = clf.predict_proba(X_valid)

# Calculate f1-score and mean squared error on validation set
f1 = f1_score(y_valid, y_pred_valid_proba.argmax(axis=1), average='macro')
mse = log_loss(y_valid, y_pred_valid_proba, labels=clf.classes_)
print(f"Valid f1-score: {f1}")
print(f"Valid mse: {mse}")
