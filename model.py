import pickle
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

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

def data_for_model(df):
    '''
    Inputs: Model DataFrame
    Outputs: Vectors for model
    '''

    games_up_to_2018_season = games_up_to_2018_season_filter(df)
    season2018 = season2018_filter(df)

    Xy_train = games_up_to_2018_season[['W', 'Wp', 'ppg', 'pApg', 'FGp',
        '3Pp', 'FTp', 'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg',
        'PFpg', 'sos', 'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2',
        'G0', 'G1', 'G2', 'G3', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp',
        'OPFTp', 'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
        'OPTOpg', 'OPPFpg', 'OPsos', 'OPexp_factor', 'C0', 'C1', 'C2',
        'F0', 'F1', 'F2', 'G0', 'G1', 'G2', 'G3']]

    Xy_test = season2018[['W', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp',
        'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
        'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2', 'G0', 'G1', 'G2',
        'G3', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp', 'OPFTp',
        'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg', 'OPTOpg',
        'OPPFpg', 'OPsos', 'OPexp_factor', 'C0', 'C1', 'C2', 'F0', 'F1',
        'F2', 'G0', 'G1', 'G2', 'G3']]

    '''Set up features and targets'''
    X_train = Xy_train.iloc[:, 1:].as_matrix()
    y_train = Xy_train.iloc[:, 0].as_matrix()
    X_test = Xy_test.iloc[:, 1:].as_matrix()
    y_test = Xy_test.iloc[:, 0].as_matrix()

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
    lr_model = LogisticRegression()

    lr_model.fit(X_train, y_train)

    y_hat = lr_model.predict(X_test)
    score = metrics.accuracy_score(y_test, y_hat)

    label30 = y_test[:30]
    pred30 = y_hat[:30]

    for i in zip(label30, pred30):
        print(i)

    print(score)
    return score

if __name__ == '__main__':

    data = pd.read_pickle('model_data/gamelog_5_exp_clust.pkl')

    X_train, y_train, X_test, y_test = data_for_model(data)

    lr_model(X_train, y_train, X_test, y_test)
