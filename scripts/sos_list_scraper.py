import os
import pandas as pd
import pickle
import requests
import time

from bs4 import BeautifulSoup
from datetime import date

from scraping_utils import school_name_transform, check_for_file, read_seasons

def clean_team(row):
    if "NCAA" in row['school']:
        row["school"] = row['school'].replace("NCAA", "").strip()
    return row

def sos_csv_creator(seasons, output_dir="0_scraped_data"):
    """
    Arguments:
        seasons (list): season years
        output_dir (str): dir to save scraped data

    Output: DataFrame of all games
    """
    
    sos_base_url = "https://www.sports-reference.com/cbb/seasons/{season}-school-stats.html#basic_school_stats::none"

    dir_files = os.listdir(output_dir)
    # print(dir_files)

    for season in seasons:

        sos_df = pd.DataFrame()

        season_filename = f"sos_list{season}.csv"

        if check_for_file(directory=output_dir, filename=season_filename):
            continue

        url = sos_base_url.format(season=season)

        """Read season school stats"""
        df = pd.read_html(url)[0]

        """Transform"""

        # Remove double Headers
        dub_header = df.columns.tolist()
        cols = [col[1].lower() for col in dub_header]
        df.columns = cols

        # Pick needed columns
        df = df[['school', 'sos']]

        # Add season column
        df['season'] = season

        # Remove divider rows
        cond1 = (df['school'] != 'Overall')
        cond2 = (df['sos'] != 'Overall')
        cond3 = (df['school'] != 'School')
        df = df[cond1 & cond2 & cond3]
        df.reset_index(inplace=True, level=None)
        df = df.drop(['index'], axis=1)

        # Remove NCAA from team name
        df = df.apply(clean_team, axis=1)

        # Update School Names
        df['school-format'] = df['school'].apply(school_name_transform)

        # iImpute mean sos for NaNs
        mean_sos = pd.to_numeric(df[df['sos'].notnull()]["sos"]).mean()
        df = df.fillna(mean_sos)

        # Save DataFrame
        filename = f"{output_dir}/{season_filename}"
        print(f"Saving: {filename}")
        df.to_csv(filename)

        time.sleep(15)

if __name__ == "__main__":

    # seasons = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    seasons = read_seasons(seasons_path='seasons_list.txt')
    # windows = [5]

    """Get strength of schedule and team list data"""
    sos_csv_creator(seasons)