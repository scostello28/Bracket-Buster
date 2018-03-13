import pandas as pd
import numpy as np

# teamsurl = 'https://www.sports-reference.com/cbb/schools/michigan-state/2018-gamelogs.html#sgl-basic::none'

teams = ['michigan-state']

def df_creator(team):
    '''
    INPUT: list of teams (formatted as in url)
    OUTPUT: DataFrame of all games
    '''
    teams_df = {}
    for team in teams:

        url = 'https://www.sports-reference.com/cbb/schools/{}/2018-gamelogs.html#sgl-basic::none'.format(team)

        '''Read team gamelog'''
        df = pd.read_html(url)[0]

        '''remove oppenent columns'''
        df = df.iloc[:, 0:23]

        '''Remove divider rows'''
        df = df.drop(df.index[[20,21]])

        '''Remove Double Row headers'''
        dubcols = df.columns.tolist()
        cols = [col[1] for col in dubcols]
        df.columns = cols

        '''Rename Columns'''
        newcols = ['G', 'Date', 'Blank', 'Opp', 'W', 'Pts', 'PtsA', 'FG', 'FGA',
                   'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'RB',
                   'AST', 'STL', 'BLK', 'TO', 'PF']
        df.columns = newcols

        '''Only take the first charcter in W field then map to 0's and 1's'''
        df['W'] = df['W'].astype(str).str[0]
        df['W'] = df['W'].map({'W': 1, 'L': 0})

        '''Create win precentage and rolling average Features'''
        df['Ws'] = df['W'].cumsum(axis=0)
        df['Wp'] = df['Ws'].astype(int) / df['G'].astype(int)
        df['ppg'] = df['Pts'].rolling(window=5,center=False).mean()
        df['pApg'] = df['PtsA'].rolling(window=5,center=False).mean()
        df['FGp'] = df['FG%'].rolling(window=5,center=False).mean()
        df['3Pg'] = df['3P%'].rolling(window=5,center=False).mean()
        df['FTp'] = df['FT%'].rolling(window=5,center=False).mean()
        df['ORBpg'] = df['ORB'].rolling(window=5,center=False).mean()
        df['RBpg'] = df['RB'].rolling(window=5,center=False).mean()
        df['ASTpg'] = df['AST'].rolling(window=5,center=False).mean()
        df['STLpg'] = df['STL'].rolling(window=5,center=False).mean()
        df['BLKpg'] = df['BLK'].rolling(window=5,center=False).mean()
        df['TOpg'] = df['TO'].rolling(window=5,center=False).mean()
        df['PFpg'] = df['PF'].rolling(window=5,center=False).mean()

        '''Remove columns after rolling ave calcs'''
        df = df.drop(['G', 'Date', 'Blank', 'Pts', 'PtsA', 'FG', 'FGA', 'FG%',
                      '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'RB',
                      'AST', 'STL', 'BLK', 'TO', 'PF'], axis=1)

        '''Drop NaN rows before rolling averages can be calc'd'''
        df = df.dropna()

        '''Add Team Column'''
        df['Tm'] = team
