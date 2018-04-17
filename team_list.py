import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import time
import pickle




def sos_csv_creator(seasons):
    '''
    Inputs:
        team = team (formatted as in url)
        season = season year

    Output: DataFrame of all games
    '''
    sos_df = pd.DataFrame()

    for season in seasons:

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

        '''Add school-format column'''
        df['school-format'] = df['school']

        '''Add season column'''
        df['season'] = season

        '''Update School Names'''
        df['school-format'] = df['school-format'].apply(school_name_transform)

        '''Remove divider rows'''
        df = df[df['school'] != 'Overall']
        df = df[df['school'] != 'School']
        df.reset_index(inplace=True, level=None)
        df = df.drop(['index'], axis=1)

        '''Transform to dict'''
        df.to_csv('sos/sos_list{}.csv'.format(season))

        time.sleep(15)


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

    '''Add school-format column'''
    df['school-format'] = df['school']

    '''Add season column'''
    df['season'] = season

    '''Update School Names'''
    df['school-format'] = df['school-format'].apply(school_name_transform)

    '''Remove divider rows'''
    df = df[(df['school'] != 'overall') & (df['school'] != 'school')]
    df.reset_index(inplace=True, level=None)
    df = df.drop(['index'], axis=1)

    '''Transform to dict'''
    df.to_csv('sos_list.csv')


def school_name_transform(school_name):
    school_name = school_name.lower()
    school_name = school_name.replace(" & ", " ")
    school_name = school_name.replace("&", "")
    school_name = school_name.replace("ncaa", "")
    school_name = school_name.strip()
    school_name = school_name.replace(" ", "-")
    school_name = school_name.replace("(", "")
    school_name = school_name.replace(")", "")
    school_name = school_name.replace(".", "")
    school_name = school_name.replace("'", "")

    if school_name == 'siu-edwardsville':
        school_name = 'southern-illinois-edwardsville'
    elif school_name == 'vmi':
        school_name = 'virginia-military-institute'
    elif school_name == 'uc-davis':
        school_name = 'california-davis'
    elif school_name == 'uc-irvine':
        school_name = 'california-irvine'
    elif school_name == 'uc-riverside':
        school_name = 'california-riverside'
    elif school_name == 'uc-santa-barbara':
        school_name = 'california-santa-barbara'
    elif school_name == 'university-of-california':
        school_name = 'california'
    elif school_name == 'louisiana':
        school_name = 'louisiana-lafayette'
    elif school_name == 'texas-rio-grande-valley':
        school_name = 'texas-pan-american'

    return school_name


if __name__ == '__main__':
    # filepath = 'team_list/sos_team_list_2018_final.csv'
    seasons = [2014, 2015, 2016, 2017, 2018]
    # teams = team_list(filepath)
    sos_csv_creator(seasons)
