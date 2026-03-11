import pandas as pd
import pickle

from scraping_utils import check_for_file, read_seasons


def gamelog_experience_cluster_merge(gamelog_df, experience_df, cluster_df, season):
    '''
    INPUT: Gamelog DataFrame and experience DataFrame
    OUTPUT: DataFrame with matching IDs merged to same row (1 game per row!)
    '''
    output_dir = "3_model_data"
    output_file_name = f"gamelog_exp_clust-{season}.pkl"

    if check_for_file(directory=output_dir, filename=output_file_name):
        return

    '''Generate ID for merge'''
    gamelog_df = gamelog_df.apply(gamelog_ID, axis=1)
    experience_df = experience_df.apply(ID, axis=1)
    cluster_df = cluster_df.apply(ID, axis=1)

    '''Drop Season columns generated with ID creation'''
    gamelog_df.drop(['Season'], axis=1, inplace=True)
    experience_df.drop(['Season', 'Team'], axis=1, inplace=True)
    cluster_df.drop(['Season', 'Team'], axis=1, inplace=True)

    '''merge experience DataFrame into gamelog DataFrame'''
    df = gamelog_df.merge(experience_df, on='ID', how='left').merge(cluster_df, on='ID', how='left')

    '''Add Unique ID for Matchup Merge'''
    df = df.apply(matchup_unique_id, axis=1)

    '''Add cumulative conditional count column'''
    df['count'] = df.groupby('ID').cumcount() + 1

    '''Create separate dataframes for 1st and 2nd instances of games'''
    df1 = df[df['count'] == 1]
    df2 = df[df['count'] == 2]

    '''Drop unneeded columns from 2nd game instance DataFrame and
    rename te prepare for pending left merge'''
    df2 = df2.drop(['Date', 'Opp', 'W', 'GameType', 'Ws', 'matchup', 'count'], axis=1)
    g2cols = df2.columns.tolist()
    OPcols = ['OP{}'.format(col) if col != 'ID'  else col for col in g2cols]
    df2.columns = OPcols

    '''Merge games instance DataFrames'''
    df = pd.merge(df1, df2, how='left', on='ID')

    '''Drop redundant Opp column and any games where there is no data
    for oppenent'''
    df = df.drop(['Ws', 'Opp', 'count', 'ID', 'count', 'matchup'], axis=1) #'just_date',
    df = df.dropna()

    filepath = f"{output_dir}/{output_file_name}"
    print(filepath)
    df.to_pickle(filepath)

def final_stats_experience_cluster_merge(gamelog_df, experience_df, cluster_df, season):
    """
    Merges Gamelog DataFrame, experience and cluster DataFrames to get full final season data for matchup prediction
    
    Parameters
    ----------
    gamelog_df (Pandas DataFrame):
    experience_df (Pandas DataFrame):
    cluster_df (Pandas DataFrame):
    season (int):

    Returns
    -------
        df (Pandas DataFrame): final stats dataframe with gamelog, clustering and experience 
    """
    output_dir = "3_model_data"
    output_file_name = f"season{season}_final_stats.pkl"

    if check_for_file(directory=output_dir, filename=output_file_name):
        return

    # Generate ID for merge
    gamelog_df = gamelog_df.apply(gamelog_ID, axis=1)
    experience_df = experience_df.apply(ID, axis=1)
    cluster_df = cluster_df.apply(ID, axis=1)

    # Drop Season columns generated with ID creation
    gamelog_df.drop(['Season'], axis=1, inplace=True)
    experience_df.drop(['Season', 'Team'], axis=1, inplace=True)
    cluster_df.drop(['Season', 'Team'], axis=1, inplace=True)

    # merge experience DataFrame into gamelog DataFrame
    df = gamelog_df.merge(experience_df, on='ID', how='left').merge(cluster_df, on='ID', how='left')

    filepath = f"{output_dir}/{output_file_name}"
    print(filepath)
    df.to_pickle(filepath)

def matchup_unique_id(row):
    '''
    Create matchup and ID rows
    '''
    row['matchup'] = ",".join(sorted([row['Tm'], row['Opp']]))
    row['ID'] = '{},{}'.format(row['matchup'], row['Date'])
    return row

def gamelog_ID(row):
    '''
    ID generator used to merge gamelog with experience and cluster DataFrames
    '''
    row['Season'] = row['GameType'][-4:]
    row['ID'] = ",".join([row['Tm'], str(row['Season'])])
    return row

def ID(row):
    '''
    ID generator used to merge experience and cluster DataFrames with gamelog
    '''
    row['ID'] = ",".join([row['Team'], str(row['Season'])])
    return row



if __name__ == "__main__":

    transformed_data = "1_transformed_data"
    full_season_data = "2_full_season_data"

    season = read_seasons(seasons_path='seasons_list.txt')[-1]

    # Read in gamelog, current year final stats, experience and cluster data
    final_stats_df = pd.read_pickle(f"{transformed_data}/season_{season}_gamelog_final_stats_data.pkl")
    gamelog_df = pd.read_pickle(f"{full_season_data}/season_full-{season}_gamelog_stats_data.pkl")
    team_experience_df = pd.read_pickle(f"{full_season_data}/team_experience-{season}.pkl")
    team_clusters_df = pd.read_pickle(f"{full_season_data}/team_clusters-{season}.pkl")

    # Matchups for modeling from Gamelog, year final stats, experience and cluster data.
    gamelog_experience_cluster_merge(gamelog_df, team_experience_df, team_clusters_df, season=season)
    final_stats_experience_cluster_merge(final_stats_df, team_experience_df, team_clusters_df, season=season)
