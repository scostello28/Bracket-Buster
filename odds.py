import pandas as pd
import numpy as np
from fancyimpute import KNN


''' Read in Data'''

odds2018 = 'odds_data/ncaa_basketball_2017-18.xlsx'
odds2017 = 'odds_data/ncaa_basketball_2016-17.xlsx'
odds2016 = 'odds_data/ncaa_basketball_2015-16.xlsx'
odds2015 = 'odds_data/ncaa_basketball_2014-15.xlsx'
odds2014 = 'odds_data/ncaa_basketball_2013-14.xlsx'
odds2013 = 'odds_data/ncaa_basketball_2012-13.xlsx'

odds2018_df = pd.read_excel(odds2018, header=0)
odds2017_df = pd.read_excel(odds2017, header=0)
odds2016_df = pd.read_excel(odds2016, header=0)
odds2015_df = pd.read_excel(odds2015, header=0)
odds2014_df = pd.read_excel(odds2014, header=0)
odds2013_df = pd.read_excel(odds2013, header=0)




def create_odds_team_name_csv(df):
    '''Create Team Name csv'''
    teams_df = df.Team.value_counts()
    teams_df = pd.DataFrame(teams_df)
    teams_df.to_csv('new_odds_teams.csv')

def odds_teams_dict(filepath):
    '''
    Create dictionary of school names and formatted school names for mapping
    '''
    team_names = pd.read_csv(filepath)
    team_names = team_names[['Teams', 'school']]
    team_dict = {}
    schools = team_names['Teams'].tolist()
    schools_format = team_names['school'].tolist()
    for school, schform in zip(schools, schools_format):
        team_dict[school] = schform
    return team_dict

def update_team_names(df):
    df['Team'] = df['Team'].map(odds_teams_dict(odds_teams_lookup_filepath))
    return df

def string_split(df):
    '''Used in impute data function to split string data into separate df'''
    string_df = df[['VH', 'Team']]
    df = df.drop(['VH', 'Team'], axis=1)
    return string_df, df

def string_to_nan(row):
    '''Used in impute_data funciton to force strings in numeric df to NaNs'''
    row = pd.to_numeric(row, errors='coerce')
    return row

def impute_data(df):
    '''
    Input: DataFrame
    Output: DataFrame with imputted missing values
    '''

    # Split out string columns into separate df
    string_df, df = string_split(df)

    # save col names
    string_df_cols = string_df.columns.tolist()
    df_cols = df.columns.tolist()

    # Convert strings to NaNs
    df = df.apply(string_to_nan, axis=1)

    #impute NaNs in df
    X = df.values
    X_filled = KNN(k=3, verbose=False).complete(X)
    df = pd.DataFrame(X_filled, columns=df_cols)
    df = pd.merge(df, string_df, how='left', left_index=True, right_index=True)
    return df

def prob(row):
    '''calc probability from ML'''
    if row['ML'] < 0:
        row['p'] = int(row['ML']) / int((row['ML']) - 100)
    elif row['ML'] > 0:
        row['p'] = 100 / int((row['ML']) + 100)
    return row

def spread(row):
    if row['p'] <= .5:
        row['spread'] = int(25 * row['p'] + -12)
    else:
        row['spread'] = int(-25 * row['p'] + 13)
    return row

def outcome(row):
    '''Adds vegas prediction, actual spread and actual W features'''
    if row['ML'] < 0:
        row['vegas'] = 1
    else:
        row['vegas'] = 0

    row['actual_spread'] = row['Final'] - row['Final_OP']

    if row['actual_spread'] > 0:
        row['W_odds'] = 1
    else:
        row['W_odds'] = 0

    return row

def date(row):
    '''Updates date format to prepare for unique ID generation'''
    row['Date'] = str(int(row['Date']))
    row['month'] = int(row['Date'][:2])
    row['day'] = int(row['Date'][-2:])
    row['Date'] = '{}-{}-{}'.format(str(row['Season']), str(row['day']), str(row['month']))
    return row

def matchups(df, season):

    '''
    Input: DataFrame and season
    Output: Home and visiting teams matched up in rows with features added
    '''

    # Drop uneeded columns
    df = df.drop(['1st', '2H', '2nd'], axis=1)

    # Add probability of winning column
    df = df.apply(prob, axis=1)

    # One hot encode VH column for counting
    df['VHohe'] = df['VH'].map({'V': 1, 'H': 0})

    # Create count column to use as merge ID
    df['count'] = df.groupby('VHohe').cumcount() + 1

    # Split df in to visitor and home team dfs
    df_v = df[df['VH'] == 'V']
    df_h = df[df['VH'] == 'H']

    # update column names for visitors df
    v_cols = df_v.columns.tolist()
    v_cols = ['{}_OP'.format(col) if col != 'count' else col for col in v_cols]
    df_v.columns = v_cols

    # Merge on count
    df = pd.merge(df_h, df_v, how='left', on='count')

    # Add Season
    df['Season'] = season

    # Add outcome
    df = df.apply(outcome, axis=1)

    # spread
    df = df.apply(spread, axis=1)

    # Update date format
    df = df.apply(date, axis=1)

    # Drop uneeded columns
    df = df.drop(['Rot', 'VH', 'VH_OP', 'Date_OP', 'Rot_OP', 'Open', 'Close',
                  'Open_OP', 'Close_OP', 'month', 'day', 'count'], axis=1)


    return df



def set_up_odds_data(df_list, season_list=[2018, 2017, 2016, 2015, 2014]):
    odds_df = pd.DataFrame()
    for df in df_list:
        df = update_team_names(df)
        df = impute_data(df)
        df = matchups(df, 2018)
        df = df.apply(outcome, axis=1)
        odds_df = odds_df.append(df, ignore_index=True)
        odds_df.dropna(inplace=True)
    odds_df.to_pickle('data/odds_data.pkl')

if __name__ == '__main__':

    # '''Create Team Name csv'''
    # create_odds_team_name_csv(df)

    '''Read in csv with team names mapped and create dict'''
    odds_teams_lookup_filepath = 'odds_teams_lookup.csv'

    odds_dfs = [odds2018_df, odds2017_df, odds2016_df, odds2015_df, odds2014_df]

    set_up_odds_data(odds_dfs)
