import pickle
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def games_up_to_2018_tourney_filter(df):
    '''Filter for games up to 2018 season'''
    notourney2018 = (df['GameType'] != 'tourney2018')
    games_up_to_2018_tourney = df[notourney2018]
    return games_up_to_2018_tourney

def tourney2018_filter(df):
    '''Filter for games up to 2018 tourney'''
    tourney2018cond = (df['GameType'] == 'tourney2018')
    tourney2018 = df[tourney2018cond]
    return tourney2018

def data_for_model(df, clusters=True, odds=False):
    '''
    Inputs: Model DataFrame
    Outputs: Vectors for model
    '''

    games_up_to_2018_tourney = games_up_to_2018_tourney_filter(df)

    if clusters and odds:
        Xy = games_up_to_2018_tourney[['W', 'Wp', 'ppg', 'pApg', 'FGp',
            '3Pp', 'FTp', 'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg',
            'PFpg', 'sos', 'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2',
            'G0', 'G1', 'G2', 'G3', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp',
            'OPFTp', 'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
            'OPTOpg', 'OPPFpg', 'OPsos', 'OPexp_factor', 'OPC0', 'OPC1', 'OPC2',
            'OPF0', 'OPF1', 'OPF2', 'OPG0', 'OPG1', 'OPG2', 'OPG3', 'final_p']]

    elif clusters and not odds:
        Xy = games_up_to_2018_tourney[['W', 'Wp', 'ppg', 'pApg', 'FGp',
            '3Pp', 'FTp', 'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg',
            'PFpg', 'sos', 'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2',
            'G0', 'G1', 'G2', 'G3', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp',
            'OPFTp', 'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
            'OPTOpg', 'OPPFpg', 'OPsos', 'OPexp_factor', 'OPC0', 'OPC1', 'OPC2',
            'OPF0', 'OPF1', 'OPF2', 'OPG0', 'OPG1', 'OPG2', 'OPG3']]
    else:
        Xy = games_up_to_2018_tourney[['W', 'Wp', 'ppg', 'pApg', 'FGp',
            '3Pp', 'FTp', 'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg',
            'PFpg', 'sos', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp',
            'OPFTp', 'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
            'OPTOpg', 'OPPFpg', 'OPsos']]

    return Xy

def set_up_data(data_df):
    '''Set up features and targets'''
    X = data_df.iloc[:, 1:].values
    y = data_df.iloc[:, 0].values

    '''Balance classes'''
    X, y = SMOTE().fit_sample(X, y)

    return X, y


def lr_model(X, y):
    '''
    Set up logistic regession pipeline.
    Input: data matricies
    Output: fit model
    '''
    lr_pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                ('model', LogisticRegression(C=0.1, penalty='l1'))])

    lr_pipeline.fit(X, y)

    filename = "fit_models/lr_fit_model_no_clust.pkl"
    with open(filename, 'wb') as f:
        # Write the model to a file.
        pickle.dump(lr_pipeline, f)

def rf_model(X, y):
    '''
    Set up Random Forest Classification pipeline.
    Input: data matricies
    Output: fit model
    '''
    rf_pipeline = Pipeline(steps=[('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                    n_estimators=530, min_samples_leaf=4,
                    min_samples_split=3, max_features='sqrt'))])

    rf_pipeline.fit(X, y)

    filename = "fit_models/rf_fit_model_no_clust.pkl"
    with open(filename, 'wb') as f:
        # Write the model to a file.
        pickle.dump(rf_pipeline, f)

def gb_model(X, y):
    '''
    Set up Random Gradient Boosting Classification pipeline.
    Input: data matricies
    Output: fit model
    '''
    gb_pipeline = Pipeline(steps=[('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(
                learning_rate=0.1, loss='exponential', max_depth=2,
                max_features=None, min_samples_leaf=2, min_samples_split=2,
                n_estimators=100, subsample=0.5))])

    gb_pipeline.fit(X, y)

    filename = "fit_models/gb_fit_model_no_clust.pkl"
    with open(filename, 'wb') as f:
        # Write the model to a file.
        pickle.dump(gb_pipeline, f)


if __name__ == '__main__':

    data = pd.read_pickle('final_model_data/gamelog_exp_clust.pkl')
    no_clust_data = pd.read_pickle('final_model_data/gamelog_exp_clust.pkl')
    odds_data = pd.read_pickle('final_model_data/gamelog_exp_clust_odds.pkl')

    Xy_data = data_for_model(data, clusters=True, odds=False)
    Xy_data_no_clust = data_for_model(no_clust_data, clusters=False, odds=False)
    Xy_data_odds = data_for_model(odds_data, clusters=True, odds=True)


    X, y = set_up_data(Xy_data)
    X_no_clust, y_no_clust = set_up_data(Xy_data_no_clust)
    X_odds, y_odds = set_up_data(Xy_data_odds)

    # print('Data with No Odds')
    # lr_model(X, y)
    # rf_model(X, y)
    # gb_model(X, y)

    # print('No Clusters or odds')
    # lr_model(X_no_clust, y_no_clust)
    # rf_model(X_no_clust, y_no_clust)
    # gb_model(X_no_clust, y_no_clust)

    # print('With Odds')
    # lr_model(X_odds, y_odds)
    # rf_model(X_odds, y_odds)
    # gb_model(X_odds, y_odds)
