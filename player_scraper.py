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


def player_df_creator(teams, seasons):
    '''
    Inputs:
        team = team (formatted as in url)
        season = season year

    Output: DataFrame of all games
    '''
    player_df = pd.DataFrame()

    for season in seasons:

        for team in teams:

            url = 'https://www.sports-reference.com/cbb/schools/{}/{}.html#per_poss::none'.format(team, season)

            '''Read team gamelog'''
            df = pd.read_html(url)[2]

            '''Add df to games_df'''
            player_df = player_df.append(df, ignore_index=True)

        time.sleep(15)

    return player_df


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

# def filter_out_reserves(df):
#     df = df[df['MP'] > np.percentile(df['MP'], 50)]
#     return df

if __name__ == '__main__':
    teams = team_list(team_names_filepath)
    seasons = [2014, 2015, 2016, 2017, 2018]
    player_stats_scraper(teams, seasons)
    # palyer_df = filter_out_reserves(player_df)
