import pandas as pd
import numpy as np
import pickle

# teamsurl = 'https://www.sports-reference.com/cbb/schools/michigan-state/2018-gamelogs.html#sgl-basic::none'

team_names_sos_filepath = 'team_list/sos_team_list_2018_final.csv'

def team_list(filepath):
    '''
    Create dictionary of school names and formatted school names for mapping
    '''
    team_names = pd.read_csv(filepath)
    school_list = team_names['School_format'].tolist()
    return school_list

def teams_dict(filepath):
    '''
    Create dictionary of school names and formatted school names for mapping
    '''
    team_names = pd.read_csv(filepath)
    team_names = team_names[['School', 'School_format']]
    team_dict = {}
    schools = team_names['School'].tolist()
    schools_format = team_names['School_format'].tolist()
    for school, schform in zip(schools, schools_format):
        team_dict[school] = schform
    return team_dict

def sos_dict(filepath):
    '''
    Create dictionary of school names and strengthof schedule for mapping
    '''
    team_sos = pd.read_csv(filepath)
    team_sos = team_sos[['School_format', 'SOS']]
    sos_dict = {}
    schools = team_sos['School_format'].tolist()
    sos = team_sos['SOS'].tolist()
    for school, sos in zip(schools, sos):
        sos_dict[school] = sos
    return sos_dict


def df_creator(teams, season):
    '''
    INPUTs:
        teams = list of teams (formatted as in url)
        season = season year

    OUTPUT: DataFrame of all games
    '''
    games_df = pd.DataFrame()

    for team in teams:

        url = 'https://www.sports-reference.com/cbb/schools/{}/{}-gamelogs.html#sgl-basic::none'.format(team, season)

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

        '''reformat Opp column strings'''
        df['Opp'] = df['Opp'].map(teams_dict(team_names_sos_filepath))

        '''Only take the first charcter in W field then map to 0's and 1's'''
        df['W'] = df['W'].astype(str).str[0]
        df['W'] = df['W'].map({'W': 1, 'L': 0})

        '''Create win precentage and rolling average Features'''
        df['Ws'] = df['W'].cumsum(axis=0)
        df['Wp'] = df['Ws'].astype(int) / df['G'].astype(int)
        df['ppg'] = df['Pts'].rolling(window=5,center=False).mean()
        df['pApg'] = df['PtsA'].rolling(window=5,center=False).mean()
        df['FGp'] = df['FG%'].rolling(window=5,center=False).mean()
        df['3Pp'] = df['3P%'].rolling(window=5,center=False).mean()
        df['FTp'] = df['FT%'].rolling(window=5,center=False).mean()
        df['ORBpg'] = df['ORB'].rolling(window=5,center=False).mean()
        df['RBpg'] = df['RB'].rolling(window=5,center=False).mean()
        df['ASTpg'] = df['AST'].rolling(window=5,center=False).mean()
        df['STLpg'] = df['STL'].rolling(window=5,center=False).mean()
        df['BLKpg'] = df['BLK'].rolling(window=5,center=False).mean()
        df['TOpg'] = df['TO'].rolling(window=5,center=False).mean()
        df['PFpg'] = df['PF'].rolling(window=5,center=False).mean()

        '''Remove columns after rolling ave calcs'''
        df = df.drop(['G', 'Blank', 'Pts', 'PtsA', 'FG', 'FGA', 'FG%',
                      '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'RB',
                      'AST', 'STL', 'BLK', 'TO', 'PF'], axis=1)

        '''Drop NaN rows before rolling averages can be calc'd'''
        df = df.dropna()

        '''Add Team Column'''
        df['Tm'] = team

        '''Add SOS columns'''
        df['sos'] = df['Tm'].map(sos_dict(team_names_sos_filepath))

        '''Add df to games_df'''
        games_df = games_df.append(df.iloc[-1], ignore_index=True)

    return games_df

def save_to_pkl(df, year):
    df.to_pickle('finalstats_{}.pkl'.format(year))

if __name__ == '__main__':
    filepath = 'team_list/sos_team_list_2018_final.csv'
    year = 2016
    teams = team_list(filepath)
    games = df_creator(teams, year)
    save_to_pkl(games, year)
