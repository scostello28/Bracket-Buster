import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


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

def set_up_data_for_model(df, model_data='cluster'):
    '''
    Inputs: Model DataFrame and DataFrame Version (gamelogs, experience, cluster)
    Outputs: Vectors for model
    '''

    games_up_to_2018_season = games_up_to_2018_season_filter(df)
    season2018 = season2018_filter(df)

    if model_data == 'gamelogs':
        Xy_train = games_up_to_2018_season[['W', 'Wp', 'ppg', 'pApg', 'FGp',
            '3Pp', 'FTp', 'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg',
            'PFpg', 'sos', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp', 'OPFTp',
            'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg', 'OPTOpg',
            'OPPFpg', 'OPsos']]

        Xy_test = season2018[['W', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp',
            'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp', 'OPFTp', 'OPORBpg',
            'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg', 'OPTOpg', 'OPPFpg',
            'OPsos']]

    elif model_data == 'experience':
        Xy_train = games_up_to_2018_season[['W', 'Wp', 'ppg', 'pApg', 'FGp',
            '3Pp', 'FTp', 'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg',
            'PFpg', 'sos', 'exp_factor', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp',
            'OP3Pp', 'OPFTp', 'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg',
            'OPBLKpg', 'OPTOpg', 'OPPFpg', 'OPsos', 'OPexp_factor']]

        Xy_test = season2018[['W', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp',
            'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'exp_factor', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp', 'OPFTp',
            'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg', 'OPTOpg',
            'OPPFpg', 'OPsos' 'OPexp_factor']]

    elif model_data == 'cluster':
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

    # Set up features and targets
    X_train = Xy_train.iloc[:, 1:].as_matrix()
    y_train = Xy_train.iloc[:, 0].as_matrix()
    X_test = Xy_test.iloc[:, 1:].as_matrix()
    y_test = Xy_test.iloc[:, 0].as_matrix()

    return X_train, y_train, X_test, y_test

def pipe(X_train, y_train, X_test, y_test):
    '''
    Set up logistic regession pipeline.
    '''
    pipeline = Pipeline(steps=['Standard Scaler', StandardScaler(),
                               'Logistic Regression', LogisticRegression()])
    pipeline.fit(X_train)
    pipeline.transform(X_test)

    predicitons = pipeline.predict(X_test)
    score = pipeline.score(X_test, y_test)

if __name__ == '__main__':

    '''
    Read in model data.
    '''

    '''
    Matchups for modeling from Gamelog (all rolling averages) data.
    '''

    # gamelog_roll2 = pd.read_pickle('model_data/gamelogs_2.pkl')
    # gamelog_roll3 = pd.read_pickle('model_data/gamelogs_3.pkl')
    # gamelog_roll4 = pd.read_pickle('model_data/gamelogs_4.pkl')
    # gamelog_roll5 = pd.read_pickle('model_data/gamelogs_5.pkl')
    # gamelog_roll6 = pd.read_pickle('model_data/gamelogs_6.pkl')
    gamelog_roll7 = pd.read_pickle('model_data/gamelogs_7.pkl')

    '''
    Matchups for modeling from Gamelog (all rolling averages) and experience data.
    '''

    # gamelog_roll2_exp = pd.read_pickle('model_data/gamelog_2_exp.pkl')
    # gamelog_roll3_exp = pd.read_pickle('model_data/gamelog_3_exp.pkl')
    # gamelog_roll4_exp = pd.read_pickle('model_data/gamelog_4_exp.pkl')
    # gamelog_roll5_exp = pd.read_pickle('model_data/gamelog_5_exp.pkl')
    # gamelog_roll6_exp = pd.read_pickle('model_data/gamelog_6_exp.pkl')
    gamelog_roll7_exp = pd.read_pickle('model_data/gamelog_7_exp.pkl')

    '''
    Matchups for modeling from Gamelog (all rolling averages), experience and cluster data.
    '''

    # gamelog_roll2_exp_cluster = pd.read_pickle('model_data/gamelog_2_exp_clust.pkl')
    # gamelog_roll3_exp_cluster = pd.read_pickle('model_data/gamelog_3_exp_clust.pkl')
    # gamelog_roll4_exp_cluster = pd.read_pickle('model_data/gamelog_4_exp_clust.pkl')
    # gamelog_roll5_exp_cluster = pd.read_pickle('model_data/gamelog_5_exp_clust.pkl')
    # gamelog_roll6_exp_cluster = pd.read_pickle('model_data/gamelog_6_exp_clust.pkl')
    gamelog_roll7_exp_cluster = pd.read_pickle('model_data/gamelog_7_exp_clust.pkl')

    # print(set_up_data_for_model(gamelog_roll7_exp_cluster, model_data='cluster'))
