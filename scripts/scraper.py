import pandas as pd
import pickle
import requests
import time

from bs4 import BeautifulSoup
from datetime import date

from scraping_utils import school_name_transform


'''sos_csv_creator needs to be run if this file is not already created'''
team_names_filepath = '0_scraped_data/sos_list2021.csv'
sos_filepath = '0_scraped_data/sos_list'
teams = team_list(team_names_filepath)


'''Season date boundaries'''
season2013start: date = date(2012,4,1)
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

season2019start = date(2018,4,5)
season2019end = date(2019,3,17)
tourney2019start = date(2019,3,19)
tourney2019end = date(2019,4,10)

season2020start = date(2019,4,11)
season2020end = date(2020,3,16)
tourney2020start = date(2020,3,17)
tourney2020end = date(2020,4,8)

season2021start = date(2020,4,9)
season2021end = date(2021,3,16)
tourney2021start = date(2021,3,17)
tourney2021end = date(2021,4,2)


# def school_name_transform(school_name):
#     school_name = school_name.lower()
#     school_name = school_name.replace(" & ", " ")
#     school_name = school_name.replace("&", "")
#     school_name = school_name.replace("ncaa", "")
#     school_name = school_name.strip()
#     school_name = school_name.replace(" ", "-")
#     school_name = school_name.replace("(", "")
#     school_name = school_name.replace(")", "")
#     school_name = school_name.replace(".", "")
#     school_name = school_name.replace("'", "")

#     if school_name == 'siu-edwardsville':
#         school_name = 'southern-illinois-edwardsville'
#     elif school_name == 'vmi':
#         school_name = 'virginia-military-institute'
#     elif school_name == 'uc-davis':
#         school_name = 'california-davis'
#     elif school_name == 'uc-irvine':
#         school_name = 'california-irvine'
#     elif school_name == 'uc-riverside':
#         school_name = 'california-riverside'
#     elif school_name == 'uc-santa-barbara':
#         school_name = 'california-santa-barbara'
#     elif school_name == 'university-of-california':
#         school_name = 'california'
#     elif school_name == 'louisiana':
#         school_name = 'louisiana-lafayette'
#     elif school_name == 'texas-rio-grande-valley':
#         school_name = 'texas-pan-american'

#     return school_name


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


def gamelog_stat_transform(df, team, sos_source, window=5, lag=True):
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


def gamelog_scraper(teams, seasons, window=5, lag=True):
    '''
    Inputs:
        team = team (formatted as in url)
        season = season year

    Output: DataFrame of all gamelogs for teams over all years
    '''

    gamelogs_df = pd.DataFrame()

    for season in seasons:

        sos_dict = sos_dict_creator(sos_filepath, season)

        for team in teams:
            '''Print for progress update'''
            print('gamelog_scraper, team: {}, season: {}, window: {}'.format(team, season, window))

            '''URL for data pull'''
            url = 'https://www.sports-reference.com/cbb/schools/{}/{}-gamelogs.html#sgl-basic::none'.format(team, season)

            '''Read team gamelog'''
            df = pd.read_html(url)[0]

            '''Transform stats'''
            df = gamelog_stat_transform(df, team, sos_dict, window, lag)

            '''Add df to games_df'''
            gamelogs_df = gamelogs_df.append(df, ignore_index=True)

        time.sleep(30)

    gamelogs_df.to_pickle('scraped_data/gamelog_data_{}_game_rolling.pkl'.format(window))


def season_final_stats_scraper(teams, season, window=5, lag=False):
    '''
    Inputs:
        team = team (formatted as in url)
        season = season year

    Output: DataFrame of final stats for each team before tourney
    '''

    season_final_stats = pd.DataFrame()
    sos_dict = sos_dict_creator(sos_filepath, season)

    for team in teams:
        '''Print for progress update'''
        print('season_final_stats_scraper, team: {}, season: {}, window: {}'.format(team, season, window))

        '''URL for data pull'''
        url = 'https://www.sports-reference.com/cbb/schools/{}/{}-gamelogs.html#sgl-basic::none'.format(team, season)

        '''Read team gamelog'''
        df = pd.read_html(url)[0]

        '''Transform tats'''
        df = gamelog_stat_transform(df, team, sos_dict, window, lag)

        '''Filter out tourney games'''
        cond = (df['GameType'] == 'season{}'.format(season))

        '''Add final stats for team to df'''
        season_final_stats = season_final_stats.append(df[cond].iloc[-1], ignore_index=True)

    season_final_stats.to_pickle('data/season{}_final_stats_{}_game_rolling.pkl'.format(season, window))


def roster_scraper(teams, seasons):
    '''
    Inputs:
        team = team (formatted as in url)
        season = season year

    Output: DataFrame of all games
    '''
    roster_df = pd.DataFrame()

    for season in seasons:

        for team in teams:
            '''Print for progress update'''
            print('roster_scraper, team: {}, season: {}'.format(team, season))

            '''URL for data pull'''
            url = 'https://www.sports-reference.com/cbb/schools/{}/{}.html#roster::none'.format(team, season)

            '''Read team gamelog'''
            df = pd.read_html(url)[0]

            '''Drop Uneeded cols'''
            df = df.iloc[:, 0:5]
            df = df.drop(['#'], axis=1)

            # '''Drop NaNs cols'''
            # df = df.dropna(axis=0, how='any')

            '''Map Class to numeric values'''
            df['Class'] = df['Class'].map({'FR': 1, 'SO': 2, 'JR': 3, 'SR': 4})

            '''Add Team col'''
            df['Team'] = team

            '''Add Season col'''
            df['Season'] = season

            '''Add df to games_df'''
            roster_df = roster_df.append(df, ignore_index=True)

        time.sleep(30)

    roster_df.to_pickle('scraped_data/roster_data.pkl')


def player_per100_scraper(teams, seasons):
    '''
    Inputs:
        team = team (formatted as in url)
        season = season year

    Output: DataFrame of all games
    '''
    player_per100_df = pd.DataFrame()

    for season in seasons:

        for team in teams:
            '''Print for progress update'''
            print('per100_scraper, team: {}, season: {}'.format(team, season))

            '''URL for data pull'''
            url = 'https://www.sports-reference.com/cbb/schools/{}/{}.html#per_poss::none'.format(team, season)

            # Extract html from player page
            req = requests.get(url).text

            # Create soup object form html
            soup = BeautifulSoup(req, 'html.parser')

            # Extract placeholder classes
            placeholders = soup.find_all('div', {'class': 'placeholder'})

            for x in placeholders:
                # Get elements after placeholder and combine into one string
                comment = ''.join(x.next_siblings)

                # Parse comment back into soup object
                soup_comment = BeautifulSoup(comment, 'html.parser')

                # Extract correct table from soup object using 'id' attribute
                tables = soup_comment.find_all('table', attrs={"id":"per_poss"})

                # Iterate tables
                for tag in tables:
                    # Turn table from html to pandas DataFrame
                    df = pd.read_html(tag.prettify())[0]

                    # Extract a player's stats from their most recent college season
                    table = df.iloc[:, :]

                    # Add Team Column
                    table['Team'] = team
                    table['Season'] = season

                    # Add individual player stats to full per_poss DataFrame
                    player_per100_df = player_per100_df.append(table).reset_index()

                    # Filter out irrelevant columns
                    player_per100_df = player_per100_df[['Player', 'G', 'GS', 'MP',
                    'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 'FT',
                    'FTA', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
                    'ORtg', 'DRtg', 'Team', 'Season']]

        time.sleep(30)

    player_per100_df.to_pickle('scraped_data/player_per100_data.pkl')


def player_roster_merger(player_pkl, roster_pkl):
    '''
    Input: 2 pickled dataframes with different player data
    Output: Saves new merged dateframe to pickle file
    '''

    '''Read in data'''
    player_df = pd.read_pickle(player_pkl)
    roster_df = pd.read_pickle(roster_pkl)

    '''Drop NaN rows and reserve players'''
    roster_df = roster_df.dropna(axis=0, how='any')

    '''Gen unique IDs for pending merge'''
    player_df = player_df.apply(player_unique_id, axis=1)
    roster_df = roster_df.apply(player_unique_id, axis=1)

    '''Drop unneeded columns'''
    roster_df = roster_df.drop(['Player', 'Team', 'Season'], axis=1)

    '''Convert Height to interger of inches'''
    roster_df = roster_df.apply(height_in, axis=1)
    roster_df = roster_df.drop(['Hf', 'Hi'], axis=1)

    '''Merge dataframes'''
    df = player_df.merge(roster_df, on='ID', how='left')

    '''Drop ID column'''
    df = df.drop(['ID'], axis=1)

    '''Map Position'''
    df = map_pos(df)

    df.to_pickle('scraped_data/player_stats.pkl')

def map_pos(df):
    pos_dict = {'G': 'G', 'PG': 'G', 'SG': 'G', 'F': 'F', 'SF': 'F', 'PF': 'F', 'C': 'C'}
    df['Pos'] = df['Pos'].map(pos_dict)
    return df

def player_unique_id(row):
    row['ID'] = ",".join([row['Player'], row['Team'], str(row['Season'])])
    return row

def height_in(row):
    row['Hf'] = int(row['Height'][0])
    row['Hi'] = int(row['Height'][1:].replace("-", ""))
    row['Height'] = row['Hf'] * 12 + row['Hi']
    return row


if __name__ == '__main__':

    seasons = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
    # windows = [5]

    '''Get full season gamelog data for all teams over all seasons'''
    # Already determinded that 5 game window works best
    # for window in windows:
    #     gamelog_scraper(teams, seasons, window=window, lag=True)

    gamelog_scraper(teams, seasons, window=5, lag=True)

    '''Get final stats gamelog data for all teams over all seasons'''
    # season_final_stats_scraper(teams, 2021, window=5, lag=False)

    '''Get roster data for all teams over all seasons'''
    # roster_scraper(teams, seasons)

    '''Get player per100 possessions data for all teams over all seasons'''
    # player_per100_scraper(teams, seasons)

    '''Merge Roster data with player per 100 stats'''
    # player_roster_merger('scraped_data/player_per100_data.pkl', 'scraped_data/roster_data.pkl')
