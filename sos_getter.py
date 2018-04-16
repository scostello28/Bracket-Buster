import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import time
import pickle


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
