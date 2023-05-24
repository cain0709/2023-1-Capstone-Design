# return dataframe shape dim-0, dim-1 int data type
import os
import sys
import pandas as pd
import numpy as np

class utils:

	def __init__(self):
		self.selected_model = ['RandomForest','ExtraTree','XGBoost','LightGBM']

		self.param_lgbm = {
			'reg_lambda': 'trial.suggest_float("reg_lambda", 1e-5, 1.0)',
			'reg_alpha': 'trial.suggest_float("reg_alpha", 1e-5, 1.0)',
			'max_depth': 'trial.suggest_int("max_depth", 4,100)',
			'colsample_bytree': 'trial.suggest_float("colsample_bytree", 0.1, 1)',
			'subsample': 'trial.suggest_loguniform("subsample", 0.5, 1)',
			'learning_rate': 'trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)',
			'n_estimators': 'trial.suggest_int("n_estimators", 100, 3000)',
			'min_child_samples': 'trial.suggest_int("min_child_samples", 5, 100)',
			'random_state' : 42
			}
		self.param_xgb = {
		        'reg_lambda': 'trial.suggest_float("reg_lambda", 1e-5, 1.0)',
		        'reg_alpha': 'trial.suggest_float("reg_alpha", 1e-5, 1.0)',
		        'max_depth': 'trial.suggest_int("max_depth", 4,100)',
		        'colsample_bytree': 'trial.suggest_float("colsample_bytree", 0.1, 1)',
		        'subsample': 'trial.suggest_float("subsample", 0.5, 1)',
		        'learning_rate': 'trial.suggest_float("learning_rate",1e-5, 1e-1)',
		        'n_estimators': 'trial.suggest_int("n_estimators", 100, 3000)',
		        'min_child_weight': 'trial.suggest_int("min_child_weight", 1, 50)',
		        'gpu_id': -1,  # CPU 사용시
			'random_state' : 42
			}
		self.param_ex = {
		        'max_depth': 'trial.suggest_int("max_depth", 10, 100)',
		        'min_samples_split': 'trial.suggest_int("min_samples_split", 2, 150)',
		        #'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 1000),
		        'n_estimators': 'trial.suggest_int("n_estimators", 100, 2500)',
		        'criterion' : 'trial.suggest_categorical("criterion", ["gini","entropy"])',
		        'bootstrap' : True,
		        'random_state': 42,
		   	}
		self.param_rf = {
		        'max_depth': 'trial.suggest_int("max_depth", 10, 100)',
		        'min_samples_split': 'trial.suggest_int("min_samples_split", 2, 150)',
		        #'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 1000),
		        'n_estimators': 'trial.suggest_int("n_estimators", 100, 2500)',
		        'criterion' : 'trial.suggest_categorical("criterion", ["gini","entropy"])',
		        'random_state': 42,
		   	}

	def return_shape(self, df):
		return df.shape[0], df.shape[1]

	#return nan check result boolean data type
	#https://suy379.tistory.com/130
	def nan_check(self, df):
		nan_check = df.isnull().sum().tolist()

		flag = False
		for value in nan_check:
			if value !=0:
				flag = True
				break
		return flag

	#https://gibles-deepmind.tistory.com/m/138 -> whether variable is numeric or categorical
	def column_type_check(self, df):
		numeric_col = df._get_numeric_data().columns.tolist()
		categorical_col = list(set(df.columns) - set(numeric_col))

		return numeric_col, categorical_col

	#class imbalance check function
	def class_imbalance_check(self, df, label_col):
		label_count_list = df[label_col].value_counts().tolist()
		flag = False
		if max(label_count_list) > min(label_count_list) * 9:
			flag = True
		return flag
	#return class_cnt
	def class_cnt_calculate(self, df, label_col):
		class_cnt = len(df[label_col].unique())
		return class_cnt

	def make_df(self, data):
	    df = pd.DataFrame(data.data, columns = data.feature_names)
	    df['target'] = data.target
	    return df

	def make_query_ML_classificaiton(self, data_path, selected_model,  label_col, class_cnt, nan_check_flag, useless_features, numeric_col, categorical_col,
		scaler_flag, class_imbalance_flag):

		Query = f'I want to make python code about machine learning classification {selected_model} model following the steps I described.\n'
		Query += f'In the code, you need to ignore warnings by using the "warinings" library, and I want to plus these library\n'
		Query += f'Please import LabelEncoder, MinMaxScaler, train_test_split, log_loss, f1_score, optuna, TPESampler on the code.\n'

		Query += f'Then, you need to use pandas library to load data from {data_path}.\n'
		Query += f'In data, The label to be predicted is {label_col}, and the label consists of {class_cnt} types.\n'
		Query += f'Also it has numeric columns such as {numeric_col}, and categoical columns such as {categorical_col}.\n'

		if useless_features is not False:
			Query += f'Since data has useless features, please drop {useless_features} columns.\n'
		
		if nan_check_flag is not False:
			Query += f'Since data has nan value, please fill nan data to zero and remove columns if each column has more than 20% missing values.' 

		if scaler_flag is not False:
			Query += f'Since scaler_flag is not false, please scale data using scikit-learn MinMaxScaler.\n'

		if class_imbalance_flag is not False:
			Query += f'Since there is a current class imbalance problem, use imblearn SMOTE library to oversample only the training data.\n'

		Query += f'First, encode categorical columns using scikit-learn labelencoder, but if there is no categorical variable, ignore this.\n'
		Query += f'Next, encode the label data (y of train or y of validation) with labelencoder.\n'
		Query += f'Then, split data into X and y.\n'

		Query += f'Also, separate 10 percent of the validation data using the train_test_split function, and set the stratify parameter of the function to True.\n' 
		Query += f'In these code, set all SEED values (and random_state parameters) to 42. \n'

		if selected_model == "RandomForest":
			Query += f'Please optimize hyperparameters using optuna library, and refer to this dictionary {self.param_rf} and write the objective function of optuna.\n'
			Query += f'Do not use pruner and use the TPESampler.\n'
		elif selected_model == "ExtraTree":
			Query += f'Please optimize hyperparameters using optuna library, and refer to this dictionary {self.param_ex} and write the objective function of optuna.\n'
			Query += f'Do not use pruner and use the TPESampler.\n'
		elif selected_model == "XGBoost":
			Query += f'Please optimize hyperparameters using optuna library, and refer to this dictionary {self.param_xgb} and write the objective function of optuna.\n'
			Query += f'Do not use pruner and use the TPESampler.\n'
		elif selected_model == "LightGBM":
			Query += f'Please optimize hyperparameters using optuna library, and refer to this dictionary {self.param_lgbm} and write the objective function of optuna.\n'
			Query += f'Do not use pruner and use the TPESampler.\n'

		Query += f'Continue with these steps, set the evaluation metric inside the objective function to macro f1-score, but fit part metric is logloss.\n'
		Query += f'After these steps, print best hyperparameters and score,valid f1 score and valid mse\n'
		
		Query += f'Finally, check your code, so that I can use code avaliable at once'

		return Query

	def make_query_ML_regression(self, data_path, selected_model, label_col, nan_check_flag, useless_features, numeric_col, categorical_col, scaler_flag):
		Query = f'I want to make python code about machine learning regression {selected_model} model following the steps I described.\n'
		Query += f'In the code, you need to ignore warnings by using the "warinings" library.\n'
		Query += f'Please import LabelEncoder, MinMaxScaler, train_test_split, f1_score, optuna, TPESampler, mean_squared_error on the code.\n'
		
		Query += f'Then, you need to use pandas library to load data from {data_path}.\n'
		Query += f'In data, The label to be predicted is {label_col}.\n'
		Query += f'Also it has numeric columns such as {numeric_col}, and categoical columns such as {categorical_col}.\n'

		if useless_features is not False:
			Query += f'Since data has useless features, please drop {useless_features} columns.\n'#useless features drop
		
		if nan_check_flag is not False:
			Query += f'Since data has nan value, please fill nan data to zero and remove columns if each column has more than 20% missing values.' #fill nan

		if scaler_flag is not False:
			Query += f'Since scaler_flag is not false, please scale data using scikit-learn MinMaxScaler.\n'


		Query += f'First, encode categorical columns using scikit-learn labelencoder, but if there is no categorical variable, ignore this.\n'
		Query += f'Next, encode the label data (y of train or y of validation) with labelencoder.\n'
		Query += f'Then, split data into X and y.\n'

		Query += f'Also, separate 10 percent of the validation data using the train_test_split function, and set the stratify parameter of the function to True.\n' 
		Query += f'In these code, set all SEED values (and random_state parameters) to 42. \n'


		if selected_model == "RandomForest":
			Query += f'Please optimize hyperparameters using optuna library, and refer to this dictionary {self.param_rf} and write the objective function of optuna.\n'
			Query += f'Do not use pruner and use the TPESampler.\n'
		elif selected_model == "ExtraTree":
			Query += f'Please optimize hyperparameters using optuna library, and refer to this dictionary {self.param_ex} and write the objective function of optuna.\n'
			Query += f'Do not use pruner and use the TPESampler.\n'
		elif selected_model == "XGBoost":
			Query += f'Please optimize hyperparameters using optuna library, and refer to this dictionary {self.param_xgb} and write the objective function of optuna.\n'
			Query += f'Do not use pruner and use the TPESampler.\n'
		elif selected_model == "LightGBM":
			Query += f'Please optimize hyperparameters using optuna library, and refer to this dictionary {self.param_lgbm} and write the objective function of optuna.\n'
			Query += f'Do not use pruner and use the TPESampler.\n'

		Query += f'Continue with these steps, set the evaluation metric inside the objective function to Mean Square Error, but fit part metric is l2.\n'
		Query += f'After these steps, print best hyperparameters and score,valid f1 score and valid mse.\n'
		
		Query += f'Finally, check your code, so that I can use code avaliable at once'

		return Query

	def classification_process(self,path_param, useless_param, label_col_param):
		data_path = path_param
		nan_check_flag = None
		useless_features = useless_param
		numeric_col = None
		categorical_col = None
		label_col = label_col_param
		class_cnt = None
		class_imbalance_flag = None
		scaler_flag = True
		Query = []


		df = pd.read_csv(data_path)
		nan_check_flag = self.nan_check(df)
		#Drop useless columns

		if useless_features[0] != '' :
			df.drop(useless_features, axis=1 ,inplace=True)
		class_cnt = self.class_cnt_calculate(df, label_col)
		class_imbalance_flag = self.class_imbalance_check(df, label_col)
		df.drop(label_col, axis=1, inplace=True)
		numeric_col, categorical_col = self.column_type_check(df)


		for model in self.selected_model:
			if model == 'LightGBM':
				self.param_lgbm['num_class'] = class_cnt
			elif model == 'XGBoost':
				self.param_xgb['num_class'] = class_cnt
			Query.append(self.make_query_ML_classificaiton(data_path, model, label_col, class_cnt, nan_check_flag, useless_features, numeric_col, categorical_col, scaler_flag, class_imbalance_flag))

			if model == 'LightGBM':
				del self.param_lgbm['num_class']
			elif model == 'XGBoost':
				del self.param_xgb['num_class']

		return Query

	def regression_process(self, path_param, useless_param, label_col_param):
		data_path = path_param
		nan_check_flag = None
		useless_features = useless_param
		numeric_col = None
		categorical_col = None
		label_col = label_col_param
		scaler_flag = True
		Query = []
		
		label_col = 'target' #Consider multivariate regression
		nan_check_flag = self.nan_check(df)

		#Drop useless columns
		if useless_features[0] != '' :
			df.drop(useless_features, axis=1 ,inplace=True)
		df.drop(label_col, axis=1, inplace=True)
		numeric_col, categorical_col = self.column_type_check(df)

		for model in self.selected_model:
			Query.append(self.make_query_ML_regression(data_path, model, label_col, nan_check_flag, useless_features, numeric_col, categorical_col, scaler_flag))
		return Query
