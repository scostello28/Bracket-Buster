import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def games_up_to_season_filter(df, season):
    '''Filter for games up to given season'''
    notourney = (df['GameType'] != f'tourney{season}')
    noseason = (df['GameType'] != f'season{season}')
    games_up_to_season = df[notourney & noseason]
    return games_up_to_season

def season_filter(df, season):
    '''Filter for given season games'''
    season_cond = (df['GameType'] == f'season{season}')
    season = df[season_cond]
    return season

def games_up_to_tourney_filter(df, season):
    '''Filter for games up to given season tourney'''
    notourney = (df['GameType'] != f'tourney{season}')
    games_up_to_tourney = df[notourney]
    return games_up_to_tourney

def tourney_filter(df, season):
    '''Filter for given season tourney games'''
    tourney_cond = (df['GameType'] == f'tourney{season}')
    tourney = df[tourney_cond]
    return tourney

def apply_filter(df, filter, season=None):
    if season: 
        return filter(df, season)
    else: 
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


def data_for_model(df, feature_set='gamelogs', train_filter=games_up_to_season_filter, test_filter=season_filter, season=None):
    '''
    Inputs: Model DataFrame
    Outputs: train and test DataFrames
    '''

    df = post_merge_feature_selection(df, feature_set=feature_set)
    if season:
        train_df = apply_filter(df, train_filter, season)
        test_df = apply_filter(df, test_filter, season)
    else:
        train_df = apply_filter(df, train_filter)
        test_df = apply_filter(df, test_filter)

    train_df = train_df.drop(['GameType'], axis=1)
    test_df = test_df.drop(['GameType'], axis=1)

    return train_df, test_df


def set_up_data(train_df, test_df, bracket=False):
    '''Set up features and targets'''
    X_train = train_df.iloc[:, 1:].values
    y_train = train_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values
    y_test = test_df.iloc[:, 0].values

    '''Balance classes'''
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    '''Standardize data'''
    scale = StandardScaler()
    scale.fit(X_train)
    X_train = scale.transform(X_train)

    if not bracket:
        X_test = scale.transform(X_test)

        return X_train, y_train, X_test, y_test
    else:
        return X_train, y_train
