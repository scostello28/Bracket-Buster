import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.externals import joblib

from filters import games_up_to_2018_season_filter, season2018_filter, data_for_model, set_up_data



def gridsearch(model, params, data):
	'''
	Gridsearch for optimal model params
	'''
	X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]
	model = GridSearchCV(model(), param_grid=params, scoring='accuracy', n_jobs=-1, cv=5, verbose=1)
	model.fit(X_train, y_train)
	best_params = model.best_params_
	score = model.best_estimator_.score(X_test, y_test)

	# file = open(“model_best_params.txt”,”w”)

	return model

def logistic_regression_grid_search(data):
	'''Logistic Regression Gridsearch'''

	lr_penalty = ['l2', 'l1']
	lr_C = list(np.arange(.1, 1.0, .1))

	lrparams = {'penalty': lr_penalty,
			  	'C': lr_C}

	lrgs = gridsearch(LogisticRegression, lrparams, data)

	best_params = lrgs.best_params_

	filename = "optimized_model_params/lr_best_params.txt"

	with open(filename, "w") as text_file:
		print("{}".format(best_params), file=text_file)

	print('Best score: {:.4f}'.format(lrgs.best_score_))
	print('Best params: {}'.format(lrgs.best_params_))

def random_forest_grid_search(data):
	'''Random Forest Gridsearch'''
	rf_n_estimators = list(range(400, 600, 10))
	rf_max_depth = [None]
	rf_min_samples_split = [3]
	rf_min_samples_leaf = [2, 3, 4, 5]
	rf_min_weight_fraction_leaf = [0.0]
	rf_max_features = ['sqrt']
	rf_max_leaf_nodes = [None]
	rf_min_impurity_decrease = [0.0]
	rf_min_impurity_split = [None]

	rfparams = {'n_estimators': rf_n_estimators,
	            'max_depth': rf_max_depth,
	            'min_samples_split': rf_min_samples_split,
	            'min_samples_leaf': rf_min_samples_leaf,
	            'min_weight_fraction_leaf': rf_min_weight_fraction_leaf,
	            'max_features': rf_max_features,
	            'max_leaf_nodes': rf_max_leaf_nodes,
	            'min_impurity_decrease': rf_min_impurity_decrease,
	            'min_impurity_split': rf_min_impurity_split}

	rfgs = gridsearch(RandomForestClassifier, rfparams, data)

	best_params = rfgs.best_params_

	filename = "optimized_model_params/rf_best_params.txt"

	with open(filename, "w") as text_file:
		print("{}".format(best_params), file=text_file)

	print('Best score: {:.4f}'.format(rfgs.best_score_))
	print('Best params: {}'.format(rfgs.best_params_))

def gradient_boosting_grid_search(data):
	'''Gradient Boosting Gridsearch'''
	gb_loss = ['deviance', 'exponential']
	gb_learning_rate = [0.05, 0.075, 0.1]
	gb_n_estimators = [50, 100, 200]
	gb_subsample = [0.5, 1.0]
	gb_min_samples_split = [2]
	gb_min_samples_leaf = [2]
	gb_max_depth = [2]
	gb_max_features = [None]

	gbparams = {'loss': gb_loss,
	          	'learning_rate': gb_learning_rate,
	            'n_estimators': gb_n_estimators,
	            'subsample': gb_subsample,
	            'min_samples_split': gb_min_samples_split,
	            'min_samples_leaf': gb_min_samples_leaf ,
	            'max_depth': gb_max_depth,
	            'max_features': gb_max_features}

	gbgs = gridsearch(GradientBoostingClassifier, gbparams, data)

	best_params = gbgs.best_params_

	filename = "optimized_model_params/gb_best_params.txt"

	with open(filename, "w") as text_file:
		print("{}".format(best_params), file=text_file)

	print('Best score: {:.4f}'.format(gbgs.best_score_))
	print('Best params: {}'.format(gbgs.best_params_))

def SVC_grid_search(data):
	'''SVC Gridsearch'''

	svc_C = [0.2, 0.4, 0.6, 0.8, 1.0]
	svc_kernel = ['linear', 'poly', 'rbf', 'sigmoid']
	svc_degree = [2, 3, 4, 5]

	svcparams = {'C': svc_C,
	          	'kernel': svc_kernel,
	            'degree': svc_degree}

	svcgs = gridsearch(SVC, svcparams, data)

	best_params = svcgs.best_params_

	filename = "optimized_model_params/svc_best_params.txt"

	with open(filename, "w") as text_file:
		print("{}".format(best_params), file=text_file)

	print('Best score: {:.4f}'.format(svcgs.best_score_))
	print('Best params: {}'.format(svcgs.best_params_))

if __name__ == '__main__':

	'''Read in model data.'''
	data_df = pd.read_pickle('2_model_data/gamelog_5_exp_clust.pkl')
	train_df, test_df = data_for_model(data_df, odds=False)
	X_train, y_train, X_test, y_test = set_up_data(train_df, test_df)
	data = (X_train, y_train, X_test, y_test)

	# logistic_regression_grid_search(data)
	# random_forest_grid_search(data)
	# gradient_boosting_grid_search(data)
	# SVC_grid_search(data)

	gb_model = gradient_boosting_grid_search(data)

	gb_model_feature_imports = gb_model.feature_importances_

	print(gb_model_feature_imports)
