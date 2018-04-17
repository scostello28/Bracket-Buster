import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import time
import pickle

'''Team list does all this!!!!!!!!!! DELETE!!!!!!!!!!'''

team_names_sos_filepath = 'team_list/sos_team_list_2018_final.csv'

def sos_df_creator(teams, seasons, window=5, lag=True):
    '''
    Inputs:
        team = team (formatted as in url)
        season = season year

    Output: DataFrame of all games
    '''
    games_df = pd.DataFrame()

    for season in seasons:

        for team in teams:

            url = 'https://www.sports-reference.com/cbb/schools/{}/{}-gamelogs.html#sgl-basic::none'.format(team, season)

            '''Read team gamelog'''
            df = pd.read_html(url)[0]

            df = stat_transform(df, team, window, lag)

            '''Add df to games_df'''
            games_df = games_df.append(df, ignore_index=True)

        time.sleep(15)

    return games_df

def sos_dict_creator(season):
    '''
    Inputs: season

    Output: Ditionary to match team with sos
    '''

    url = 'https://www.sports-reference.com/cbb/seasons/{}-school-stats.html#basic_school_stats::none'.format(season)

    '''Read season school stats'''
    df = pd.read_html(url)[0]

    '''Transform'''

    '''Remove double Headers'''
    dub_header = df.columns.tolist()
    cols = [col[1].lower() for col in dub_header]
    df.columns = cols

    '''Pick needed columns'''
    df = df[['school', 'sos']]

    '''Add season column'''
    df['season'] = season

    '''Update School Names'''
    df['school'] = df['school'].apply(school_name_transform)

    '''Remove divider rows'''
    df = df[(df['school'] != 'overall') & (df['school'] != 'school')]
    # df = df[df['school'] != 'overall']
    # df = df[df['school'] != 'school']
    df.reset_index(inplace=True, level=None)
    df = df.drop(['index'], axis=1)

    '''Transform to dict'''
    sos_dict = {k: float(v)for k, v in zip(df['school'], df['sos'])}

    return sos_dict
