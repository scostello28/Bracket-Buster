import pickle
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score as cvs
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def games_up_to_2018_season_filter(df):
    '''Filter for games up to 2018 season'''
    notourney2018 = (df['GameType'] != 'tourney2018')
    noseason2018 = (df['GameType'] != 'season2018')
    games_up_to_2018_season = df[notourney2018 & noseason2018]
    return games_up_to_2018_season

def season2018_filter(df):
    '''Filter for games up to 2018 season'''
    season2018cond = (df['GameType'] == 'season2018')
    season2018 = df[season2018cond]
    return season2018

def data_for_model(df, cluster=False):
    '''
    Inputs: Model DataFrame
    Outputs: Vectors for model
    '''

    games_up_to_2018_season = games_up_to_2018_season_filter(df)
    season2018 = season2018_filter(df)

    if cluster:
        Xy_train = games_up_to_2018_season[['W', 'Wp', 'ppg', 'pApg', 'FGp',
            '3Pp', 'FTp', 'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg',
            'PFpg', 'sos', 'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2',
            'G0', 'G1', 'G2', 'G3', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp',
            'OPFTp', 'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
            'OPTOpg', 'OPPFpg', 'OPsos', 'OPexp_factor', 'OPC0', 'OPC1', 'OPC2',
            'OPF0', 'OPF1', 'OPF2', 'OPG0', 'OPG1', 'OPG2', 'OPG3']]

        Xy_test = season2018[['W', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp',
            'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2', 'G0', 'G1', 'G2',
            'G3', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp', 'OPFTp',
            'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg', 'OPTOpg',
            'OPPFpg', 'OPsos', 'OPexp_factor', 'OPC0', 'OPC1', 'OPC2', 'OPF0', 'OPF1',
            'OPF2', 'OPG0', 'OPG1', 'OPG2', 'OPG3']]

    else:
        Xy_train = games_up_to_2018_season[['W', 'Wp', 'ppg', 'pApg', 'FGp',
            '3Pp', 'FTp', 'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg',
            'PFpg', 'sos', 'exp_factor', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp',
            'OPFTp', 'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
            'OPTOpg', 'OPPFpg', 'OPsos', 'OPexp_factor']]

        Xy_test = season2018[['W', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp',
            'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'exp_factor', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp', 'OPFTp',
            'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg', 'OPTOpg',
            'OPPFpg', 'OPsos', 'OPexp_factor']]

    return Xy_train, Xy_test

def set_up_data(train_df, test_df):
    '''Set up features and targets'''
    X_train = train_df.iloc[:, 1:].values
    y_train = train_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values
    y_test = test_df.iloc[:, 0].values

    '''Balance classes'''
    X_train, y_train = SMOTE().fit_sample(X_train, y_train)

    '''Standardize data'''
    scale = StandardScaler()
    scale.fit(X_train)
    X_train = scale.transform(X_train)
    X_test = scale.transform(X_test)

    return X_train, y_train, X_test, y_test


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
    # return score

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
    # odds_data = pd.read_pickle('final_model_data/gamelog_exp_clust_odds.pkl')

    train_df, test_df = data_for_model(data, cluster=False)
    cluster_train_df, cluster_test_df = data_for_model(data, cluster=True)

    X_train, y_train, X_test, y_test = set_up_data(train_df, test_df)
    X_train_cluster, y_train_cluster, X_test_cluster, y_test_cluster = set_up_data(cluster_train_df, cluster_test_df)

    print('Without Team Composition Features')
    lr_model(X_train, y_train, X_test, y_test)
    rf_model(X_train, y_train, X_test, y_test)
    gb_model(X_train, y_train, X_test, y_test)

    print('With Team Composition Features')
    lr_model(X_train_cluster, y_train_cluster, X_test_cluster, y_test_cluster)
    rf_model(X_train_cluster, y_train_cluster, X_test_cluster, y_test_cluster)
    gb_model(X_train_cluster, y_train_cluster, X_test_cluster, y_test_cluster)
