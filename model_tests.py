import pickle
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.model_selection import cross_val_score as cvs
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from filters import data_for_model, set_up_data, post_merge_feature_selection, games_up_to_2018_season_filter, season2018_filter

def lr_model(X_train, y_train, X_test, y_test):
    '''
    Set up logistic regession pipeline.
    Input: train and test matricies
    Output: model predictions and accuracy
    '''
    lr_model = LogisticRegression(C=0.1, penalty='l1')

    lr_model.fit(X_train, y_train)

    cv_score = np.mean(cvs(lr_model, X_train, y_train, scoring='accuracy', cv=5, n_jobs=-1))
    y_hat = lr_model.predict(X_test)
    score = metrics.accuracy_score(y_test, y_hat)

    print('LR CV Accuracy: {:.2f}'.format(cv_score))
    print('LR Test Accuracy: {:.2f}'.format(score))

def rf_model(X_train, y_train, X_test, y_test):
    '''
    Set up logistic regession pipeline.
    Input: train and test matricies
    Output: model predictions and accuracy
    '''
    rf_model = RandomForestClassifier(n_estimators=500, min_samples_leaf=4,
    min_samples_split=3, max_features='sqrt')

    rf_model.fit(X_train, y_train)

    cv_score = np.mean(cvs(rf_model, X_train, y_train, scoring='accuracy', cv=5, n_jobs=-1))
    y_hat = rf_model.predict(X_test)
    score = metrics.accuracy_score(y_test, y_hat)

    print('RF CV Accuracy: {:.2f}'.format(cv_score))
    print('RF Test Accuracy: {:.2f}'.format(score))

def gb_model(X_train, y_train, X_test, y_test):
    '''
    Set up logistic regession pipeline.
    Input: train and test matricies
    Output: model predictions and accuracy
    '''
    gb_model = GradientBoostingClassifier(learning_rate=0.1, loss='exponential', max_depth=2, max_features=None, min_samples_leaf=2, min_samples_split=2, n_estimators=100, subsample=0.5)

    gb_model.fit(X_train, y_train)

    cv_score = np.mean(cvs(gb_model, X_train, y_train, scoring='accuracy', cv=5, n_jobs=-1))
    y_hat = gb_model.predict(X_test)
    score = metrics.accuracy_score(y_test, y_hat)

    print('GB CV Accuracy: {:.2f}'.format(cv_score))
    print('GB Test Accuracy: {:.2f}'.format(score))

if __name__ == '__main__':

    data = pd.read_pickle('3_final_model_data/gamelog_exp_tcf.pkl')

    '''Set up data for model'''
    train_df, test_df = data_for_model(data, feature_set='gamelogs', train_filter=games_up_to_2018_season_filter, test_filter=season2018_filter)
    train_df_exp_tcf, test_df_exp_tcf = data_for_model(data, feature_set='exp_tcf', train_filter=games_up_to_2018_season_filter, test_filter=season2018_filter)

    X_train, y_train, X_test, y_test = set_up_data(train_df, test_df)
    X_train_exp_tcf, y_train_exp_tcf, X_test_exp_tcf, y_test_exp_tcf = set_up_data(train_df_exp_tcf, test_df_exp_tcf)

    print('w/o TCF')
    lr_model(X_train, y_train, X_test, y_test)
    rf_model(X_train, y_train, X_test, y_test)
    gb_model(X_train, y_train, X_test, y_test)

    print('w/ TCF')
    lr_model(X_train_exp_tcf, y_train_exp_tcf, X_test_exp_tcf, y_test_exp_tcf)
    rf_model(X_train_exp_tcf, y_train_exp_tcf, X_test_exp_tcf, y_test_exp_tcf)
    gb_model(X_train_exp_tcf, y_train_exp_tcf, X_test_exp_tcf, y_test_exp_tcf)
