import pandas as pd
import pickle

from scraping_utils import school_name_transform, team_list, teams_dict, sos_dict_creator, check_for_file, read_seasons

def lag_columns(df, cols_to_shift):
    """
    Input: DataFrame
    Output: DataFrame with stats lagged so matchup stats included in matchup stats rolling average
    """
    for col in cols_to_shift:
        new_col = '{}_shifted'.format(col)
        df[new_col] = df[col].shift(1)
    df = df.drop(cols_to_shift, axis=1)
    column_names = df.columns.tolist()
    new_column_names = [col.replace('_shifted', '') for col in column_names]
    df.columns = new_column_names
    df = df.dropna()
    return df


def calc_gamelog_stats(df, window=5, lag=True):
    """
    INPUTs:
        df = dataframe created from html pull
        team to add team column

    OUTPUT: DataFrame of all games with clean and transformed data
    """

    """Create win precentage and rolling average Features"""
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

    """Remove columns after rolling ave calcs"""
    df = df.drop(['G', 'Pts', 'PtsA', 'FG', 'FGA', 'FG%',
                  '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'RB',
                  'AST', 'STL', 'BLK', 'TO', 'PF'], axis=1)

    """Drop NaN rows before rolling averages can be calc'd"""
    df = df.dropna()


    cols_to_shift = ['Ws', 'Wp','ppg', 'pApg', 'FGp', '3Pp', 'FTp',
       'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'Tm']

    if lag:
        df = lag_columns(df, cols_to_shift)

    return df


def gamelog_stats_transform(seasons, source_dir, output_dir, window=5, lag=True, verbose=False):
    """
    Inputs:
        team = team (formatted as in url)
        season = season year

    Output: DataFrame of all gamelogs for teams over all years
    """

    for season in seasons:
        
        season_df = pd.DataFrame()

        # Get teams list for season
        team_names_filepath = f"0_scraped_data/sos_list{season}.csv"
        teams = team_list(team_names_filepath)

        gamelog_season_filename = f"season_{season}_gamelog_data.pkl"
        stats_season_filename = f"season_{season}_gamelog_stats_data.pkl"

        if check_for_file(directory=output_dir, filename=stats_season_filename):
            continue
        
        print(f"Transforming {season} stats")

        df = pd.read_pickle(source_dir + "/" + gamelog_season_filename)

        # in 2023 some of the stats included NaN
        df = df[df['W'].notna()].copy()

        try:
            if verbose:
                """Print for progress update"""
                print(f"gamelog stats transform, team: {team}, season: {season}")

            teams_df = pd.DataFrame()
            
            for team in teams:
                # Filter to team
                team_cond = (df['Tm'] == team)
                team_df = df[team_cond].copy()
                # apply stats transform
                team_df = calc_gamelog_stats(team_df)

                """Add team_df to teams_df"""
                teams_df = teams_df.append(team_df, ignore_index=True)

        except Exception as e:
            print(e)
            raise e
                
        """Add teams_df to season_df"""
        season_df = season_df.append(teams_df, ignore_index=True)
        
        print(f"Saving: {stats_season_filename}")
        season_df.to_pickle(f"{output_dir}/{stats_season_filename}")

    
def gamelog_final_stats(seasons, source_dir, output_dir, verbose=False):
    """
    Inputs:
        team = team (formatted as in url)
        season = season year

    Output: DataFrame of all gamelogs for teams over all years
    """

    for season in seasons:
        
        season_df = pd.DataFrame()

        gamelog_season_filename = f"season_{season}_gamelog_stats_data.pkl"
        stats_season_filename = f"season_{season}_gamelog_final_stats_data.pkl"
      
        if check_for_file(directory=output_dir, filename=stats_season_filename):
            continue
        
        print(f"Transforming {season} stats")

        df = pd.read_pickle(source_dir + "/" + gamelog_season_filename)

        if verbose:
            """Print for progress update"""
            print(f"gamelog final stats, team: {team}, season: {season}")

        try:
            teams_df = pd.DataFrame()

            for team in set(df.Tm.values):
                # Filter to team and season
                team_cond = (df['Tm'] == team)
                season_cond = (df['GameType'] == 'season{}'.format(season))
                team_df = df[team_cond & season_cond].copy()
                print(team)
                # filter to final game of season for team
                team_df = team_df.iloc[-1, :]

                """Add team_df to teams_df"""
                teams_df = teams_df.append(team_df, ignore_index=True)

        except Exception as e:
            print(e)
            raise e
                
        """Add teams_df to season_df"""
        season_df = season_df.append(teams_df, ignore_index=True)
        
        print(f"Saving: {stats_season_filename}")
        season_df.to_pickle(f"{output_dir}/{stats_season_filename}")


if __name__ == '__main__':

    # Dependencies:
    # - sos_csv_creator
    # - gamelog_scraper

    seasons = read_seasons(seasons_path='seasons_list.txt')

    # Get gamelog stats
    gamelog_stats_transform(seasons, source_dir="0_scraped_data", output_dir="1_transformed_data")

    # Get final season stats
    gamelog_final_stats(seasons, source_dir="1_transformed_data", output_dir="1_transformed_data")