Solution:


import warnings

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna

warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('C:/Users/na018/Desktop/capstone/2023-1-Capstone-Design/iris.csv')
    
# Drop 'Id' column
data.drop(['Id'], axis=1, inplace=True)

# Encode categorical columns
if data.select_dtypes(include=['object']).shape[1] > 0:
    le = LabelEncoder()
    data[data.select_dtypes(include=['object']).columns] = \
        data[data.select_dtypes(include=['object']).columns].apply(le.fit_transform)

# Prepare data
X = data.drop(['Species'], axis=1)
y = data['Species']

# Encode label data
le_y = LabelEncoder()
le_y.fit(y)
y = le_y.transform(y)

# Scale data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Define objective function
def objective(trial):
    # Define hyperparameters and their search spaces
    params = {'reg_lambda': trial.suggest_float("reg_lambda", 1e-5, 1.0),
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
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    # Train model and evaluate on validation set
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    score = model.evals_result()['validation_0']['mlogloss'][model.best_iteration-1]
    
    return score

# Optimize hyperparameters
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)

# Print optimized hyperparameters
print(study.best_trial.params)


Libraries:


- pandas
- sklearn
- xgboost
- optuna
