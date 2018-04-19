import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import time
import pickle
import requests
from bs4 import BeautifulSoup

url = 'https://www.sports-reference.com/cbb/schools/abilene-christian/2018.html#per_poss::none'
team_names_filepath = 'sos/sos_list2018.csv'

def team_list(filepath):
    '''
    Create dictionary of school names and formatted school names for mapping
    '''
    team_names = pd.read_csv(filepath)
    school_list = team_names['school-format'].tolist()
    return school_list


def roster_df_creator(teams, seasons):
    '''
    Inputs:
        team = team (formatted as in url)
        season = season year

    Output: DataFrame of all games
    '''
    roster_df = pd.DataFrame()

    for season in seasons:

        for team in teams:
            print(team, season)
            url = 'https://www.sports-reference.com/cbb/schools/{}/{}.html#roster::none'.format(team, season)

            '''Read team gamelog'''
            df = pd.read_html(url)[0]

            '''Drop Uneeded cols'''
            df = df.iloc[:, 0:5]
            df = df.drop(['#'], axis=1)

            '''Drop NaNsa cols'''
            df = df.dropna(axis=0, how='any')

            '''Map Class to upper=1 and lower=0'''
            df['Class'] = df['Class'].map({'FR': 0, 'SO': 1, 'JR': 1, 'SR': 1})

            '''Add Team col'''
            df['Team'] = team

            '''Add Season col'''
            df['Season'] = season

            '''Add df to games_df'''
            roster_df = roster_df.append(df, ignore_index=True)

        time.sleep(30)

    roster_df.to_pickle('players/rosters.pkl')


def player_stats_scraper(teams, seasons):
    '''
    Inputs:
        team = team (formatted as in url)
        season = season year

    Output: DataFrame of all games
    '''
    player_per100_df = pd.DataFrame()

    for season in seasons:

        for team in teams:

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

    player_per100_df.to_pickle('players/player_per100.pkl')

def player_roster_merger(player_pkl, roster_pkl):
    '''
    Input: 2 pickled dataframes with different player data
    Output: Saves new merged dateframe to pickle file
    '''

    '''Read in data'''
    player_df = pd.read_pickle(player_pkl)
    roster_df = pd.read_pickle(roster_pkl)

    '''Gen unique IDs for pending merge'''
    player_df = player_df.apply(unique_id, axis=1)
    roster_df = roster_df.apply(unique_id, axis=1)

    '''Drop unneeded columns'''
    roster_df = roster_df.drop(['Player', 'Team', 'Season'], axis=1)

    '''Convert Height to interger of inches'''
    roster_df = roster_df.apply(height_in, axis=1)
    roster_df = roster_df.drop(['Hf', 'Hi'], axis=1)

    '''Merge dataframes'''
    df = player_df.merge(roster_df, on='ID', how='left')

    '''Drop ID column'''
    df = df.drop(['ID'], axis=1)

    '''Drop NaN rows and reserve players'''
    df = df.dropna(axis=0, how='any')
    df = df[df['MP'] > np.percentile(df['MP'], 25)]

    df.to_pickle('players/player_stats.pkl')

'''Row Functions'''

def unique_id(row):
    row['ID'] = ",".join([row['Player'], row['Team'], str(row['Season'])])
    return row

def height_in(row):
    row['Hf'] = int(row['Height'][0])
    row['Hi'] = int(row['Height'][1:].replace("-", ""))
    row['Height'] = row['Hf'] * 12 + row['Hi']
    return row


if __name__ == '__main__':
    teams = team_list(team_names_filepath)
    seasons = [2014, 2015, 2016, 2017, 2018]

    '''Output pkl file with each players stats for all years'''
    # player_stats_scraper(teams, seasons)

    '''Output pkl file with rosters for all teams for each year'''
    # roster_df_creator(teams, seasons)

    '''Merge player_stats_scraper and roster_df_creator outputs'''
    player_roster_merger('players/player_per100.pkl', 'players/rosters.pkl')
