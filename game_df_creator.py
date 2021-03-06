import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import time
import pickle
import pdb

'''sos_getter needs to be run if this file does not exist'''
team_names_filepath = 'sos/sos_list2018.csv'
sos_filepath = 'sos/sos_list'

def team_list(filepath):
    '''
    Create dictionary of school names and formatted school names for mapping
    '''
    team_names = pd.read_csv(filepath)
    school_list = team_names['school-format'].tolist()
    return school_list


def teams_dict(filepath):
    '''
    Create dictionary of school names and formatted school names for mapping
    '''
    team_names = pd.read_csv(filepath)
    team_names = team_names[['school', 'school-format']]
    team_dict = {}
    schools = team_names['school'].tolist()
    schools_format = team_names['school-format'].tolist()
    for school, schform in zip(schools, schools_format):
        team_dict[school] = schform
    return team_dict


def sos_dict_creator(filepath, season):
    '''
    Create dictionary of school names and strength of schedule for mapping
    '''
    filepath = filepath + str(season) + '.csv'
    team_sos = pd.read_csv(filepath)
    team_sos = team_sos[['school-format', 'sos']]
    sos_dict = {}
    schools = team_sos['school-format'].tolist()
    sos = team_sos['sos'].tolist()
    for school, sos in zip(schools, sos):
        sos_dict[school] = sos
    return sos_dict


def add_game_type(row):
    '''
    Create Column for tourney games
    '''

    if row['just_date'] >= tourney2014start and row['just_date'] <= tourney2014end:
        row['GameType'] = 'tourney2014'

    elif row['just_date'] >= season2014start and row['just_date'] <= season2014end:
        row['GameType'] = 'season2014'

    elif row['just_date'] >= tourney2015start and row['just_date'] <= tourney2015end:
        row['GameType'] = 'tourney2015'

    elif row['just_date'] >= season2015start and row['just_date'] <= season2015end:
        row['GameType'] = 'season2015'

    elif row['just_date'] >= tourney2016start and row['just_date'] <= tourney2016end:
        row['GameType'] = 'tourney2016'

    elif row['just_date'] >= season2016start and row['just_date'] <= season2016end:
        row['GameType'] = 'season2016'

    elif row['just_date'] >= tourney2017start and row['just_date'] <= tourney2017end:
        row['GameType'] = 'tourney2017'

    elif row['just_date'] >= season2017start and row['just_date'] <= season2017end:
        row['GameType'] = 'season2017'

    elif row['just_date'] >= tourney2018start and row['just_date'] <= tourney2018end:
        row['GameType'] = 'tourney2018'

    elif row['just_date'] >= season2018start and row['just_date'] <= season2018end:
        row['GameType'] = 'season2018'

    else:
        row['GameType'] = 'season'

    return row


def lag_columns(df, cols_to_shift):
    '''
    Input: DataFrame
    Output: DataFrame with stats lagged so matchup stats included in matchup stats rolling average
    '''
    for col in cols_to_shift:
        new_col = '{}_shifted'.format(col)
        df[new_col] = df[col].shift(1)
    df = df.drop(cols_to_shift, axis=1)
    column_names = df.columns.tolist()
    new_column_names = [col.replace('_shifted', '') for col in column_names]
    df.columns = new_column_names
    df = df.dropna()
    return df


def stat_transform(df, team, sos_source, window=5, lag=True):
    '''
    INPUTs:
        df = dataframe created from html pull
        team to add team column

    OUTPUT: DataFrame of all games with clean and transformed data
    '''

    '''remove oppenent columns'''
    df = df.iloc[:, 0:23]


    '''Remove Double Column headers'''
    dubcols = df.columns.tolist()
    cols = [col[1] for col in dubcols]
    df.columns = cols

    '''Rename Columns'''
    newcols = ['G', 'Date', 'Blank', 'Opp', 'W', 'Pts', 'PtsA', 'FG', 'FGA',
               'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'RB',
               'AST', 'STL', 'BLK', 'TO', 'PF']
    df.columns = newcols

    '''Remove divider rows'''
    df = df[(df['Date'] != 'School') & (df['Date'] != 'Date')]

    '''reformat Opponent team name column strings'''
    df['Opp'] = df['Opp'].map(teams_dict(team_names_filepath))
    # df['Opp'] = df['Opp'].apply(school_name_transform)

    '''Only take the first charcter in W field then map to 0's and 1's.
    (Ties and overtime have excess characters)'''
    df['W'] = df['W'].astype(str).str[0]
    df['W'] = df['W'].map({'W': 1, 'L': 0})

    '''Create win precentage and rolling average Features'''
    # pdb.set_trace()
    df['Ws'] = df['W'].cumsum(axis=0)
    df['Wp'] =  df['Ws'].astype(int) / df['G'].astype(int)
    df['ppg'] = df['Pts'].rolling(window=window,center=False).mean()
    df['pApg'] = df['PtsA'].rolling(window=window,center=False).mean()
    df['FGp'] = df['FG%'].rolling(window=window,center=False).mean()
    df['3Pp'] = df['3P%'].rolling(window=window,center=False).mean()
    df['FTp'] = df['FT%'].rolling(window=window,center=False).mean()
    df['ORBpg'] = df['ORB'].rolling(window=window,center=False).mean()
    df['RBpg'] = df['RB'].rolling(window=window,center=False).mean()
    df['ASTpg'] = df['AST'].rolling(window=window,center=False).mean()
    df['STLpg'] = df['STL'].rolling(window=window,center=False).mean()
    df['BLKpg'] = df['BLK'].rolling(window=window,center=False).mean()
    df['TOpg'] = df['TO'].rolling(window=window,center=False).mean()
    df['PFpg'] = df['PF'].rolling(window=window,center=False).mean()

    '''Remove columns after rolling ave calcs'''
    df = df.drop(['G', 'Blank', 'Pts', 'PtsA', 'FG', 'FGA', 'FG%',
                  '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'RB',
                  'AST', 'STL', 'BLK', 'TO', 'PF'], axis=1)

    '''Drop NaN rows before rolling averages can be calc'd'''
    df = df.dropna()

    '''Add Team Column'''
    df['Tm'] = team

    '''Add SOS columns'''
    df['sos'] = df['Tm'].map(sos_source)

    '''Add datetime formatted date without time of day (i.e. just the date)'''
    df['just_date'] = pd.to_datetime(df['Date']).dt.date

    df = df.apply(add_game_type, axis=1)

    df = df.drop(['just_date'], axis=1)

    cols_to_shift = ['Ws', 'Wp','ppg', 'pApg', 'FGp', '3Pp', 'FTp',
       'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'Tm']

    if lag:
        df = lag_columns(df, cols_to_shift)
    else:
        pass

    return df


def gen_unique_id(row):
    '''
    Create matchup and ID rows
    '''
    row['matchup'] = ",".join(sorted([row['Tm'], row['Opp']]))
    row['ID'] = '{},{}'.format(row['matchup'], row['Date'])
    return row


def everybody_merge(df):
    '''
    INPUT: DataFrame
    OUTPUT: DataFrame with matching IDs merged to same row (1 game per row!)
    '''

    '''Add cumulative conditional count column'''
    df['count'] = df.groupby('ID').cumcount() + 1

    '''Create separate dataframes for 1st and 2nd instances of games'''
    df1 = df[df['count'] == 1]
    df2 = df[df['count'] == 2]

    '''Drop unneeded columns from 2nd game instance DataFrame and
    rename te prepare for pending left merge'''
    df2 = df2.drop(['Date', 'Opp', 'W', 'GameType', 'Ws', 'matchup', 'count'], axis=1)
    g2cols = df2.columns.tolist()
    OPcols = ['OP{}'.format(col) if col != 'ID' else col for col in g2cols]
    df2.columns = OPcols

    '''Merge games instance DataFrames'''
    df = pd.merge(df1, df2, how='left', on='ID')

    '''Drop redundant Opp column and any games where there is no data
    for oppenent'''
    df = df.drop(['Date', 'Ws', 'Opp', 'count', 'ID', 'matchup', 'count', 'Tm', 'OPTm'], axis=1) #'just_date',
    df = df.dropna()

    return df


def games_df_creator(teams, seasons, window=5, lag=True):
    '''
    Inputs:
        team = team (formatted as in url)
        season = season year

    Output: DataFrame of all games
    '''

    games_df = pd.DataFrame()

    for season in seasons:

        sos_dict = sos_dict_creator(sos_filepath, season)

        for team in teams:
            print(team, season)
            url = 'https://www.sports-reference.com/cbb/schools/{}/{}-gamelogs.html#sgl-basic::none'.format(team, season)

            '''Read team gamelog'''
            df = pd.read_html(url)[0]

            df = stat_transform(df, team, sos_dict, window, lag)

            '''Add df to games_df'''
            games_df = games_df.append(df, ignore_index=True)

            # time.sleep(5)
        # '''Add SOS columns'''
        # df['sos'] = df['Tm'].map(sos_dict)

        time.sleep(40)

    return games_df


def season_final_stats(teams, season, window=5, lag=False):
    '''
    Inputs:
        team = team (formatted as in url)
        season = season year

    Output: DataFrame of final stats for each team before tourney
    '''

    season_final_stats = pd.DataFrame()
    sos_dict = sos_dict_creator(sos_filepath, season)

    for team in teams:

        url = 'https://www.sports-reference.com/cbb/schools/{}/{}-gamelogs.html#sgl-basic::none'.format(team, season)

        '''Read team gamelog'''
        df = pd.read_html(url)[0]

        df = stat_transform(df, team, sos_dict, window, lag)

        cond = (df['GameType'] == 'season{}'.format(season))

        season_final_stats = season_final_stats.append(df[cond].iloc[-1], ignore_index=True)

        # time.sleep(5)
    # '''Add SOS columns'''
    # df['sos'] = df['Tm'].map(sos_dict)

    season_final_stats.to_pickle('game_data/season{}_final_stats.pkl'.format(season))


def season_games(seasons, teams):
    '''
    Creates DataFrame for full season and tourney for given year
    Input: Season
    Output: DataFrame of all games in given year
    '''
    games = games_df_creator(teams, seasons, window=5, lag=True)
    games = games.apply(gen_unique_id, axis=1)
    games = everybody_merge(games)
    games.to_pickle('game_data/all_games.pkl')
    # games.to_pickle('game_data/games_{}.pkl'.format(season))
    return games


if __name__ == '__main__':
    teams = team_list(team_names_filepath)

    season2013start = date(2012,4,1)
    season2013end = date(2013,3,18)
    tourney2013start = date(2013,3,19)
    tourney2013end = date(2013,4,8)

    season2014start = date(2013,4,9)
    season2014end = date(2014,3,17)
    tourney2014start = date(2014,3,18)
    tourney2014end = date(2014,4,7)

    season2015start = date(2014,4,8)
    season2015end = date(2015,3,16)
    tourney2015start = date(2015,3,17)
    tourney2015end = date(2015,4,6)

    season2016start = date(2015,4,7)
    season2016end = date(2016,3,14)
    tourney2016start = date(2016,3,15)
    tourney2016end = date(2016,4,4)

    season2017start = date(2016,4,5)
    season2017end = date(2017,3,13)
    tourney2017start = date(2017,3,14)
    tourney2017end = date(2017,4,3)

    season2018start = date(2017,4,4)
    season2018end = date(2018,3,12)
    tourney2018start = date(2018,3,13)
    tourney2018end = date(2018,4,2)

    seasons = [2014, 2015, 2016, 2017, 2018]

    '''Get game date for all seasons'''
    games = season_games(seasons, teams)

    '''Remove these filters'''
    # '''Games up to 2017 tourney'''
    # games_up_to_2017_tourney = games_up_to_2017_tourney_filter(games)
    #
    # '''Games up to 2018 season'''
    # games_up_to_2018_season = games_up_to_2018_season_filter(games)
    #
    # '''up to 2018 tourney'''
    # games_up_to_2018_tourney = games_up_to_2018_tourney_filter(games)

    '''2017 season final stats'''
    season_final_stats(teams, 2017, window=5, lag=False)

    '''2018 season final stats'''
    season_final_stats(teams, 2018, window=5, lag=False)
