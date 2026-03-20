import pandas as pd
import pickle
import requests
import time

from bs4 import BeautifulSoup
from datetime import date
from urllib.error import HTTPError

from scraping_utils import school_name_transform, team_list, teams_dict, sos_dict_creator, check_for_file, read_seasons


def roster_scraper(seasons, source_dir, output_dir, verbose=False):
    '''
    Inputs:
        season = season year

    Output: DataFrame of all games
    '''

    for season in seasons:

        print(f"Scraping rosters from {season} season")

        roster_df_list = []

        # Get teams list for season
        team_names_filepath = f"{output_dir}/sos_list{season}.csv"
        teams = team_list(team_names_filepath)

        season_filename = f"roster_{season}_data.csv"

        if check_for_file(directory=output_dir, filename=season_filename):
            continue

        for team in teams:
            try:
                if verbose:
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
            
            except HTTPError as http_error:
                print(http_error)
                print(url)
                print(f"skip {season} {team}")
            except ValueError as value_error:
                print(value_error)
                print(url)
                print(f"skip {season} {team}")
            else:
                '''Add df to roster_df_list'''
                roster_df_list.append(df)
                
            time.sleep(10)

        roster_df = pd.concat(roster_df_list, ignore_index=True)

        print(f"Saving {season_filename}")
        roster_df.to_csv(f"{output_dir}/{season_filename}")

        time.sleep(30)


def player_per100_scraper(season, source_dir, output_dir):
    '''
    Inputs:
        season = season year

    Output: DataFrame of all games
    '''

    df_list = []

    team_counter = 0

    for team in teams:
        try:
            '''Print for progress update'''
            print('per100_scraper, team: {}, season: {}'.format(team, season))

            '''URL for data pull'''
            url = 'https://www.sports-reference.com/cbb/schools/{}/{}.html#per_poss'.format(team, season)
            
            # df = pd.read_html(url)[11]

            try:
                df = pd.read_html(url)[11]
            except IndexError as index_error:
                print(index_error)
                print(url)
                print("Index is out of range try alternate table index as team may not have conference stats")
                df = pd.read_html(url)[6]
            
            # Drop uneeded columns
            # df = df.drop(['Rk', 'Unnamed: 24'], axis=1)
            df = df.drop(['Rk', 'Awards'], axis=1) # 2025 fix
            
            # Add Team and Season Columns
            df['Team'] = team
            df['Season'] = season

        except HTTPError as http_error:
            print(http_error)
            print(url)
            print(f"skip {season} {team}")
        except ValueError as value_error:
            print(value_error)
            print(url)
            print(f"skip {season} {team}")
        else:
            # Add individual player stats to full per_poss DataFrame
            df_list.append(df)

        team_counter += 1
        if team_counter == 10:
            time.sleep(10)
            team_counter = 0
        else:
            time.sleep(5)

    player_per100_df = pd.concat(df_list, ignore_index=True)

    # Filter out irrelevant columns
    cols = ['Player', 'G', 'GS', 'MP',
    'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 'FT',
    'FTA', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
    'ORtg', 'DRtg', 'Team', 'Season']

    player_per100_df = player_per100_df[cols]
        
    print(f"Saving {season_filename}")
    player_per100_df.to_pickle(f'{source_dir}/{season_filename}')

    time.sleep(30)
        

def player_per100_scraper_pre_2022(season, source_dir, output_dir):
    '''
    Inputs:
        season = season year

    Output: DataFrame of all games
    '''

    df_list = []

    team_counter = 0

    for team in teams:
        try:
            '''Print for progress update'''
            print('per100_scraper, team: {}, season: {}'.format(team, season))

            '''URL for data pull'''
            url = 'https://www.sports-reference.com/cbb/schools/{}/{}.html#per_poss'.format(team, season)

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
                tables = soup_comment.find_all('table', attrs={"id":"players_per_poss"})

                # Iterate tables
                for tag in tables:
                    # Turn table from html to pandas DataFrame
                    df = pd.read_html(tag.prettify())[0]

                    # Extract a player's stats from their most recent college season
                    table = df.iloc[:, :]

                    # Add Team Column
                    table['Team'] = team
                    table['Season'] = season

        except HTTPError as http_error:
            print(http_error)
            print(url)
            print(f"skip {season} {team}")
        except ValueError as value_error:
            print(value_error)
            print(url)
            print(f"skip {season} {team}")
        else:
            # Add individual player stats to full per_poss DataFrame
            df_list.append(df)

        team_counter += 1
        if team_counter == 10:
            time.sleep(10)
            team_counter = 0
        else:
            time.sleep(5)

    player_per100_df = pd.concat(df_list, ignore_index=True)
                

    # Filter out irrelevant columns
    cols = [
        'Player', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', 
        '3PA', '3P%', 'FT', 'FTA', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 
        'PTS', 'ORtg', 'DRtg', 'Team', 'Season'
        ]

    player_per100_df = player_per100_df[cols]
        
    print(f"Saving {season_filename}")
    player_per100_df.to_pickle(f'{source_dir}/{season_filename}')

    time.sleep(30)


if __name__ == '__main__':

    # Dependencies:
    # - sos_csv_creator
    seasons = read_seasons(seasons_path='seasons_list.txt')
    source_dir="/Users/sean/Documents/bracket_buster/data/0_scraped_data"
    output_dir="/Users/sean/Documents/bracket_buster/data/0_scraped_data"

    roster_scraper(seasons, source_dir=source_dir, output_dir=output_dir, verbose=True)

    
    for season in seasons:

        player_per100_df = pd.DataFrame()

        # Get teams list for season
        team_names_filepath = f"{output_dir}/sos_list{season}.csv"
        teams = team_list(team_names_filepath)

        season_filename = f"player_per100_{season}_data.pkl"

        if check_for_file(directory=output_dir, filename=season_filename):
            continue
        if int(season) < 2022:
            player_per100_scraper_pre_2022(
                season=season, 
                source_dir=source_dir, 
                output_dir=output_dir
            )
                
        else:
            player_per100_scraper(
                season=season, 
                source_dir=source_dir, 
                output_dir=output_dir
                )
