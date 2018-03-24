import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def games_up_to_2017_tourney_filter(df):
    '''Filter for all games up to 2017 tourney'''
    notourney2018 = (df['GameType'] != 'tourney2018')
    noseason2018 = (df['GameType'] != 'season2018')
    notourney2017 = (df['GameType'] != 'tourney2017')
    games_up_to_2017_tourney = df[notourney2018 & noseason2018 & notourney2017]
    return games_up_to_2017_tourney

def set_up_train_set(df):
        '''Shuffle DataFrames'''
        df = df.sample(frac=1).reset_index(drop=True)

        Xy_train = df[['W', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
                          'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
                          'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp', 'OPFTp',
                          'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
                          'OPTOpg', 'OPPFpg', 'OPsos']]

        # Set up features and targets
        X_train = Xy_train.iloc[:, 1:].as_matrix()
        y_train = Xy_train.iloc[:, 0].as_matrix()

        return X_train, y_train

def merge(df, team1, team2):
    '''
    INPUT: DataFrame
    OUTPUT: DataFrame with matching IDs merged to same row
    '''
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

    '''Merge games instance DataFrames'''
    df1['game'] = 'game'
    df2['game'] = 'game'

    dfout = pd.merge(df1, df2, how='left', on='game')
    dfout = dfout.drop(['game', 'Tm', 'OPTm'], axis=1)

    return dfout

def game_predict(matchup, X_train, y_train):
    '''Fit model on training data'''
    lg = LogisticRegression()  # penalty='l2' as default which is Ridge
    lg.fit(X_train, y_train)
    lg_predict = lg.predict(matchup)
    lg_prob = lg.predict_proba(matchup)

    if lg_predict[0] == 0:
        print('{} loses and {} wins!'.format(team1, team2))
        print('{} has {}% chance to win.'.format(team1, int(lg_prob[0][1]*100)))
        print('{} has {}% chance to win.'.format(team2, int(lg_prob[0][0]*100)))
    else:
        print('{} wins and {} loses!'.format(team1, team2))
        print('{} has {:.0f}% chance to win.'.format(team1, lg_prob[0][1]*100))
        print('{} has {:.0f}% chance to win.'.format(team2, lg_prob[0][0]*100))

if __name__ == '__main__':
    games = pd.read_pickle('game_data/all_games.pkl')
    games = games_up_to_2017_tourney_filter(games)
    finalgames = pd.read_pickle('game_data/season2017_final_stats.pkl')
    X_train, y_train = set_up_train_set(games)
    team1 = str(input('team1: '))
    team2 = str(input('team2: '))
    matchup = merge(finalgames, team1, team2)
    game_predict(matchup, X_train, y_train)
