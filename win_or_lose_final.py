import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def merge(df, team1, team2, cluster):
    '''
    INPUT: DataFrame
    OUTPUT: DataFrame with matching IDs merged to same row
    '''
    if cluster:
        df = df[['Tm', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg', 'RBpg',
                'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos', 'exp_factor',
                'C0', 'C1', 'C2', 'F0', 'F1', 'F2', 'G0', 'G1', 'G2', 'G3']]

    else:
        df = df[['Tm', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg', 'RBpg',
            'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos']]

    '''Create separate dataframes for 1st and 2nd instances of games'''
    df1 = df.loc[df['Tm'] == team1,:]
    df2 = df.loc[df['Tm'] == team2,:]

    '''Select needed columns from 2nd instance DataFrame and
    rename te prepare for pending left merge'''
    # df2_stats = df2  #.iloc[:, 5:19]

    df2cols = df2.columns.tolist()
    OPcols = ['OP{}'.format(col) for col in df2cols]
    df2.columns = OPcols

    '''Force unique ID to merge on'''
    df1['game'] = 'game'
    df2['game'] = 'game'

    '''Merge games instance DataFrames'''
    dfout = pd.merge(df1, df2, how='left', on='game')

    '''Drop uneeded columns'''
    dfout = dfout.drop(['game', 'Tm', 'OPTm'], axis=1)

    return dfout

def game_predict(model, matchup, matchup_reversed):
    '''Predict on matchup'''
    prob = model.predict_proba(matchup)
    prob_reversed = model.predict_proba(matchup_reversed)
    team1_prob = (prob[0][1] + prob_reversed[0][0]) / 2 * 100
    team2_prob = (prob[0][0] + prob_reversed[0][1]) / 2 * 100

    '''Print results'''
    if team1_prob < team2_prob:
        print('{} wins and {} loses!'.format(team2, team1))
        print('{} has {}% chance to win.'.format(team1, int(round(team1_prob))))
        print('{} has {}% chance to win.'.format(team2, int(round(team2_prob))))
    else:
        print('{} wins and {} loses!'.format(team1, team2))
        print('{} has {}% chance to win.'.format(team1, int(round(team1_prob))))
        print('{} has {}% chance to win.'.format(team2, int(round(team2_prob))))


if __name__ == '__main__':

    finalgames2018 = pd.read_pickle('final_model_data/season2018_final_stats.pkl')
    finalgames2018_noclust = pd.read_pickle('final_model_data/season2018_final_stats_no_clust.pkl')

    lr_model = 'fit_models/lr_fit_model.pkl'
    rf_model = 'fit_models/rf_fit_model.pkl'
    gb_model = 'fit_models/gb_fit_model.pkl'
    # mlp_model = 'fit_models/mlp_fit_model.pkl'
    lr_model_no_clust = 'fit_models/lr_fit_model_no_clust.pkl'
    rf_model_no_clust = 'fit_models/rf_fit_model_no_clust.pkl'
    gb_model_no_clust = 'fit_models/gb_fit_model_no_clust.pkl'
    # mlp_model_no_clust = 'fit_models/mlp_fit_model_no_clust.pkl'

    pickled_model = lr_model
    pickled_model_no_clust = lr_model_no_clust

    with open(pickled_model, 'rb') as f:
        model = pickle.load(f)

    with open(pickled_model_no_clust, 'rb') as f_no_clust:
        model_no_clust = pickle.load(f_no_clust)

    team1 = str(input('team1: '))
    team2 = str(input('team2: '))

    matchup = merge(finalgames2018, team1, team2, cluster=True)
    matchup_reversed = merge(finalgames2018, team2, team1, cluster=True)
    game_predict(model, matchup, matchup_reversed)

    no_clust_matchup = merge(finalgames2018, team1, team2, cluster=False)
    no_clust_matchup_reversed = merge(finalgames2018, team2, team1, cluster=False)
    game_predict(model_no_clust, no_clust_matchup, no_clust_matchup_reversed)
