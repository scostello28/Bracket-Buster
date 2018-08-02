import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def games_up_to_2017_tourney_filter(df):
    '''Filter for games up to 2017 tourney'''
    notourney2018 = (df['GameType'] != 'tourney2018')
    noseason2018 = (df['GameType'] != 'season2018')
    notourney2017 = (df['GameType'] != 'tourney2017')
    games_up_to_2017_tourney = df[notourney2018 & noseason2018 & notourney2017]
    return games_up_to_2017_tourney

def games_up_to_2018_season_filter(df):
    '''Filter for games up to 2018 season'''
    notourney2018 = (df['GameType'] != 'tourney2018')
    noseason2018 = (df['GameType'] != 'season2018')
    games_up_to_2018_season = df[notourney2018 & noseason2018]
    return games_up_to_2018_season

def season2018_filter(df):
    '''Filter for 2018 season games'''
    season2018cond = (df['GameType'] == 'season2018')
    season2018 = df[season2018cond]
    return season2018

def games_up_to_2018_tourney_filter(df):
    '''Filter for games up to 2018 tourney'''
    notourney2018 = (df['GameType'] != 'tourney2018')
    games_up_to_2018_tourney = df[notourney2018]
    return games_up_to_2018_tourney

def tourney2018_filter(df):
    '''Filter for 2018 tourney games'''
    tourney2018cond = (df['GameType'] == 'tourney2018')
    tourney2018 = df[tourney2018cond]
    return tourney2018

def apply_filter(df, filter):
    return filter(df)


def pre_matchup_feature_selection(df, feature_set='gamelogs'):
    '''
    Inputs: Model DataFrame
    Outputs: DataFrame with features selected
    '''

    if feature_set == 'gamelogs':
        df = df[['W', 'Tm', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
            'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos']]

    elif feature_set == 'exp':
        df = df[['W', 'Tm', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
            'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'exp_factor']]

    elif feature_set == 'tcf':
        df = df[['W', 'Tm', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
            'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'C0', 'C1', 'C2', 'F0', 'F1', 'F2', 'G0', 'G1', 'G2', 'G3']]

    elif feature_set == 'exp_tcf':
        df = df[['W', 'Tm', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
            'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2', 'G0', 'G1',
            'G2', 'G3']]

    elif feature_set == 'odds':
        df = df[['W', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
            'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2', 'G0', 'G1',
            'G2', 'G3', 'final_p']]

    return df


def post_merge_feature_selection(df, feature_set='gamelogs'):
    '''
    Inputs: Model DataFrame
    Outputs: DataFrame with features selected
    '''

    if feature_set == 'gamelogs':
        df = df[['W', 'GameType', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
            'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp', 'OPFTp', 'OPORBpg',
            'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg', 'OPTOpg', 'OPPFpg',
            'OPsos']]

    elif feature_set == 'exp':
        df = df[['W', 'GameType', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
            'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'exp_factor', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp',
            'OPFTp', 'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
            'OPTOpg', 'OPPFpg', 'OPsos', 'OPexp_factor']]

    elif feature_set == 'tcf':
        df = df[['W', 'GameType', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
            'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'C0', 'C1', 'C2', 'F0', 'F1', 'F2', 'G0', 'G1', 'G2', 'G3',
            'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp', 'OPFTp', 'OPORBpg',
            'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg', 'OPTOpg', 'OPPFpg',
            'OPsos', 'OPC0', 'OPC1', 'OPC2', 'OPF0', 'OPF1', 'OPF2', 'OPG0',
            'OPG1', 'OPG2', 'OPG3']]

    elif feature_set == 'exp_tcf':
        df = df[['W', 'GameType', 'Wp', 'ppg', 'pApg', 'FGp',
            '3Pp', 'FTp', 'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg',
            'PFpg', 'sos', 'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2',
            'G0', 'G1', 'G2', 'G3', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp',
            'OPFTp', 'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
            'OPTOpg', 'OPPFpg', 'OPsos', 'OPexp_factor', 'OPC0', 'OPC1', 'OPC2',
            'OPF0', 'OPF1', 'OPF2', 'OPG0', 'OPG1', 'OPG2', 'OPG3']]

    elif feature_set == 'odds':
        df = df[['W', 'GameType', 'Wp', 'ppg', 'pApg', 'FGp',
            '3Pp', 'FTp', 'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg',
            'PFpg', 'sos', 'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2',
            'G0', 'G1', 'G2', 'G3', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp',
            'OPFTp', 'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
            'OPTOpg', 'OPPFpg', 'OPsos', 'OPexp_factor', 'OPC0', 'OPC1', 'OPC2',
            'OPF0', 'OPF1', 'OPF2', 'OPG0', 'OPG1', 'OPG2', 'OPG3', 'final_p']]

    return df

def data_for_model(df, feature_set='gamelogs', train_filter=games_up_to_2018_season_filter, test_filter=season2018_filter):
    '''
    Inputs: Model DataFrame
    Outputs: train and test DataFrames
    '''

    df = post_merge_feature_selection(df, feature_set=feature_set)
    train_df = apply_filter(df, train_filter)
    test_df = apply_filter(df, test_filter)

    train_df = train_df.drop(['GameType'], axis=1)
    test_df = test_df.drop(['GameType'], axis=1)

    return train_df, test_df

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

def load_model_data_for_mlp(data_df):
    '''Set up features and targets for mlp'''
    X = data_df.iloc[:, 1:].values
    y = data_df.iloc[:, 0].values
    X = X_train.astype(theano.config.floatX)
    return X, y
