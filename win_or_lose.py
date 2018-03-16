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

games = pd.read_pickle('game_data/games_four_years.pkl')
finalgames = pd.read_pickle('game_data/finalstats_2016.pkl')

'''Shuffle DataFrames'''
games = games.sample(frac=1).reset_index(drop=True)

Xy_train = games[['W', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
                  'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
                  'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp', 'OPFTp',
                  'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
                  'OPTOpg', 'OPPFpg', 'OPsos']]

# Set up features and targets
X_train = Xy_train.iloc[:, 1:].as_matrix()
y_train = Xy_train.iloc[:, 0].as_matrix()

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
    df2_stats = df2  #.iloc[:, 5:19]

    g2cols = df2_stats.columns.tolist()
    OPcols = ['OP{}'.format(col) for col in g2cols]
    df2_stats.columns = OPcols

    '''Merge games instance DataFrames'''
    df1['game'] = 'game'
    df2_stats['game'] = 'game'

    dfout = pd.merge(df1, df2_stats, how='left', on='game')
    dfout = dfout.drop(['game', 'Tm', 'OPTm', 'OPWp'], axis=1)

    return dfout  #.as_matrix()

team1 = str(input('team1: ')) #'kansas'   #'colorado'
team2 = str(input('team2: ')) #'north-carolina' #'connecticut'
matchup = merge(finalgames, team1, team2)
# print(len(matchup.columns.tolist()))
# print(len(Xy_train.columns.tolist()))

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


'''

df of final stats for each team for 2016 (and 2017) create pkls

merge by team1 and team 2
    - drop cols (Opp, etc)
    - filter to team 1 then team 2
    - merge team 2 to team 1 and rename columns

csv of teams in tourney and formatted name to reference

funtion to:
    transform two teams into format for testing

    merge by team1 and team 2
        - drop cols (Opp, etc)
        - filter to team 1 then team 2
        - merge team 2 to team 1 and rename columns


'''
