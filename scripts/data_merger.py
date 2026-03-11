import pandas as pd
import pickle

from scraping_utils import check_for_file, read_seasons


def concat_seasons(filepath, seasons, source_dir, output_dir):

    final_season = seasons[-1]
    # stop func if full season data already exists in output dir
    if check_for_file(output_dir, f"{filepath.format(f'full-{final_season}')}"):
        return 

    master_df = pd.DataFrame()

    for season in seasons:
        filename = f"{source_dir}/{filepath.format(season)}"
        print(filename)

        if filepath[-3:] == 'pkl':
            df = pd.read_pickle(filename)
        elif filepath[-3:] == 'csv':
            df = pd.read_csv(filename, index_col=0)

        master_df = master_df.append(df, ignore_index=True)

    print(f"Saving: {filepath.format(f'full-{final_season}')}")
    if filepath[-3:] == 'pkl':
        master_df.to_pickle(f"{output_dir}/{filepath.format(f'full-{final_season}')}")
    elif filepath[-3:] == 'csv':
        master_df.to_csv(f"{output_dir}/{filepath.format(f'full-{final_season}')}")
    
    

    
def map_pos(df):
    pos_dict = {'G': 'G', 'PG': 'G', 'SG': 'G', 'F': 'F', 'SF': 'F', 'PF': 'F', 'C': 'C'}
    df['Pos'] = df['Pos'].map(pos_dict)
    return df


def player_unique_id(row):
    row['ID'] = ",".join([row['Player'], row['Team'], str(row['Season'])])
    return row


def height_in(row):
    """
    TODO: this should be vectorized with t = rdf['Height'].str.split('-', expand=True) instead of row func
    """
    row['Hf'] = int(row['Height'][0])
    hi = row['Height'][1:].replace("-", "")

    # Remove and height in inches recorded as decimal
    if hi.find('.') != -1:
        hi = hi[:hi.find('.')]

    row['Hi'] = int(hi)
    row['Height'] = row['Hf'] * 12 + row['Hi']
    return row


def player_roster_merger(source_dir, output_dir, final_season):
    '''
    Input: 2 pickled dataframes with different player data
    Output: Saves new merged dateframe to pickle file
    '''

    
    player_stats_filename = f"player_stats_full-{final_season}.pkl"

    if check_for_file(directory=output_dir, filename=player_stats_filename):
        return

    player_filename = f"player_per100_full-{final_season}_data.pkl"
    roster_filename = f"roster_full-{final_season}_data.csv"

    '''Read in data'''
    player_df = pd.read_pickle(f"{source_dir}/{player_filename}")
    roster_df = pd.read_csv(f"{source_dir}/{roster_filename}", index_col=0)

    '''Drop NaN rows and reserve players'''
    # TODO: fill height in rosters so that players arent dropped if thats the only NaN
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

    print(f"Saving {player_stats_filename}")
    df.to_pickle(f"{output_dir}/{player_stats_filename}")


if __name__ == '__main__':

    # Dependencies:
    # - sos_csv_creator
    # - player_scraper

    seasons = read_seasons(seasons_path='seasons_list.txt')

    gamelog_stats_filepath = "season_{}_gamelog_stats_data.pkl"
    roster_filepath = "roster_{}_data.csv"
    player_per100_filepath = "player_per100_{}_data.pkl"

    print("Concat Gamelog Stats Data")
    concat_seasons(filepath=gamelog_stats_filepath, seasons=seasons, source_dir="1_transformed_data", output_dir="2_full_season_data")
    print("Concat Roster Data")
    concat_seasons(filepath=roster_filepath, seasons=seasons, source_dir="0_scraped_data", output_dir="2_full_season_data")
    print("Concat Player per 100 Data")
    concat_seasons(filepath=player_per100_filepath, seasons=seasons, source_dir="0_scraped_data", output_dir="2_full_season_data")

    print("player_roster_merger")
    player_roster_merger(source_dir="2_full_season_data", output_dir="2_full_season_data", final_season=seasons[-1])