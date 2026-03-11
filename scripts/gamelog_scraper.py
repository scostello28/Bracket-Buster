import pandas as pd
import pickle
import requests
import time

from bs4 import BeautifulSoup
from datetime import date
from urllib.error import HTTPError

from scraping_utils import school_name_transform, team_list, teams_dict, sos_dict_creator, check_for_file, read_seasons


def add_game_type(row):
    """
    Create Column for tourney games
    """

    """Season date boundaries"""
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
    tourney2019start = date(2019,3,18)
    tourney2019end = date(2019,4,10)

    season2020start = date(2019,4,11)
    season2020end = date(2020,3,16)
    tourney2020start = date(2020,3,17)
    tourney2020end = date(2020,4,8)

    season2021start = date(2020,4,9)
    season2021end = date(2021,3,16)
    tourney2021start = date(2021,3,17)
    tourney2021end = date(2021,4,2)

    season2022start = date(2021,4,3)
    season2022end = date(2022,3,16)
    tourney2022start = date(2022,3,17)
    tourney2022end = date(2022,4,5)

    season2023start = date(2022,4,6)
    season2023end = date(2023,3,16)
    tourney2023start = date(2023,3,15)
    tourney2023end = date(2023,4,5)

    season2024start = date(2023,4,6)
    season2024end = date(2024,3,18)
    tourney2024start = date(2024,3,19)
    tourney2024end = date(2024,4,9)

    season2025start = date(2024,4,10)
    season2025end = date(2024,3,18)
    tourney2025start = date(2025,3,19)
    tourney2025end = date(2025,4,15)

    season2026start = date(2025,4,16)
    season2026end = date(2025,3,18)
    tourney2026start = date(2026,3,19)
    tourney2026end = date(2026,4,15)


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
    
    elif row['just_date'] >= tourney2019start and row['just_date'] <= tourney2019end:
        row['GameType'] = 'tourney2019'

    elif row['just_date'] >= season2019start and row['just_date'] <= season2019end:
        row['GameType'] = 'season2019'

    elif row['just_date'] >= tourney2020start and row['just_date'] <= tourney2020end:
        row['GameType'] = 'tourney2020'

    elif row['just_date'] >= season2020start and row['just_date'] <= season2020end:
        row['GameType'] = 'season2020'

    elif row['just_date'] >= tourney2021start and row['just_date'] <= tourney2021end:
        row['GameType'] = 'tourney2021'

    elif row['just_date'] >= season2021start and row['just_date'] <= season2021end:
        row['GameType'] = 'season2021'

    elif row['just_date'] >= tourney2022start and row['just_date'] <= tourney2022end:
        row['GameType'] = 'tourney2022'

    elif row['just_date'] >= season2022start and row['just_date'] <= season2022end:
        row['GameType'] = 'season2022'

    elif row['just_date'] >= tourney2023start and row['just_date'] <= tourney2023end:
        row['GameType'] = 'tourney2023'

    elif row['just_date'] >= season2023start and row['just_date'] <= season2023end:
        row['GameType'] = 'season2023'

    elif row['just_date'] >= tourney2024start and row['just_date'] <= tourney2024end:
        row['GameType'] = 'tourney2024'

    elif row['just_date'] >= season2024start and row['just_date'] <= season2024end:
        row['GameType'] = 'season2024'

    elif row['just_date'] >= tourney2025start and row['just_date'] <= tourney2025end:
        row['GameType'] = 'tourney2025'

    elif row['just_date'] >= season2025start and row['just_date'] <= season2025end:
        row['GameType'] = 'season2025'

    elif row['just_date'] >= tourney2026start and row['just_date'] <= tourney2026end:
        row['GameType'] = 'tourney2026'

    elif row['just_date'] >= season2026start and row['just_date'] <= season2026end:
        row['GameType'] = 'season2026'

    else:
        row['GameType'] = 'season'

    return row


def home_away_map(row):
    """Clean up Home col"""
    if type(row["Home"]) == str:
        if row["Home"] == "@":
            row["Home"] = "A"
    elif type(row["Home"]) == float:
        if row["Home"] != row["Home"]:
            row["Home"] = "H"
    return row


def clean_team_gamelog(df, team, sos_season_dict):
    """
    Clean up gamelog DataFrame

    df (DataFrame): gamelog DataFrame
    team (str): teams gamelog
    """

    """remove oppenent columns"""
    df = df.iloc[:, 1:31]

    """Remove Double Column headers"""
    dubcols = df.columns.tolist()
    cols = [col[1] for col in dubcols]
    df.columns = cols

    df = df.drop(columns=['Type', 'OT', '2P', '2PA', '2P%', 'eFG%', 'TRB'])

    # Rename Columns
    newcols = ['G', 'Date', 'Home', 'Opp', 'W', 'Pts', 'PtsA', 'FG', 'FGA',
            'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'RB',
            'AST', 'STL', 'BLK', 'TO', 'PF']
    df.columns = newcols

    # Remove divider rows
    cond1 = (df['Date'] != 'Date')
    cond2 = (df['Date'].notnull())
    df = df[cond1 & cond2]

    # Clean up Home col
    df = df.apply(home_away_map, axis=1)

    """Only take the first charcter in W field then map to 0's and 1's.
    (Ties and overtime have excess characters)"""
    df['W'] = df['W'].astype(str).str[0]
    df['W'] = df['W'].map({'W': 1, 'L': 0})

    # Reformat Opponent team name column strings
    df['Opp'] = df['Opp'].apply(school_name_transform)

    """Add Team Column"""
    df['Tm'] = team

    """Add SOS columns"""
    df['sos'] = df['Tm'].map(sos_season_dict)

    """Add datetime formatted date without time of day (i.e. just the date)"""
    df['just_date'] = pd.to_datetime(df['Date']).dt.date

    df = df.apply(add_game_type, axis=1)

    df = df.drop(['just_date'], axis=1)

    return df


def gamelog_scraper(seasons, output_dir="/Users/sean/Documents/bracket_buster/data/0_scraped_data"):
    """
    Bot/Scraping/Crawler Traffic on Sports-Reference.com Sites
    https://www.sports-reference.com/bot-traffic.html

    Inputs:
        team = team (formatted as in url)
        season = season year

    Output: DataFrame of all gamelogs for teams over all years
    """

    gamelog_base_url = "https://www.sports-reference.com/cbb/schools/{team}/{season}-gamelogs.html#sgl-basic::none"

    for season in seasons:

        season_df = pd.DataFrame()

        # Get teams list for season
        team_names_filepath = f"{output_dir}/sos_list{season}.csv"
        teams = team_list(team_names_filepath)

        # Get SOS dict for season
        sos_base_filepath = f"{output_dir}/sos_list"
        sos_season_dict = sos_dict_creator(sos_base_filepath, season)

        season_filename = f"season_{season}_gamelog_data.pkl"

        if check_for_file(directory=output_dir, filename=season_filename):
            continue

        print(f"Scraping {season} gamelogs")

        for team in teams:
            try:
                """Print for progress update"""
                print(f"gamelog_scraper, team: {team}, season: {season}")

                """URL for data pull"""
                url = gamelog_base_url.format(team=team, season=season)

                """Read team gamelog"""
                df = pd.read_html(url)[0]

                """Clean gamelog data"""
                df = clean_team_gamelog(df, team, sos_season_dict)

            except HTTPError as http_error:
                print(http_error)
                print(url)
                print(f"skip {season} {team}")
            except ValueError as value_error:
                print(value_error)
                print(url)
                print(f"skip {season} {team}")
            else:
                """Add df to games_df"""
                # season_df = season_df.append(df, ignore_index=True)
                season_df = pd.concat([season_df, df], ignore_index=True)

            time.sleep(10)
        
        print(f"Saving {season_filename}")
        season_df.to_pickle(f"{output_dir}/{season_filename}")

        time.sleep(30)


if __name__ == '__main__':

    """sos_csv_creator needs to be run if this file is not already created"""

    seasons = read_seasons(seasons_path='seasons_list.txt')

    """Get full season gamelog data for all teams over all seasons"""
    gamelog_scraper(seasons)
