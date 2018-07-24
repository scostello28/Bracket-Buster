import pandas as pd
import pickle
import pdb

def gamelog_experience_cluster_merge(gamelog_df, experience_df, cluster_df, filepath):
    '''
    INPUT: Gamelog DataFrame and experience DataFrame
    OUTPUT: DataFrame with matching IDs merged to same row (1 game per row!)
    '''

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

    filepath = 'model_data/gamelog_{}_exp_clust.pkl'.format(str(window))
    final_filepath = 'final_model_data/gamelog_exp_clust.pkl'
    print(filepath)
    df.to_pickle(filepath)

def final_stats_experience_cluster_merge(gamelog_df, experience_df, cluster_df, filepath):
    '''
    INPUT: Gamelog DataFrame and experience DataFrame
    OUTPUT: DataFrame with matching IDs merged to same row (1 game per row!)
    '''

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

    # final_filepath = 'final_model_data/season2017_final_stats.pkl'
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



if __name__ == '__main__':

    '''
    Read in gamelog data with all rolling average windows, team experience data, and cluster data.
    '''
    # 5 game rolling average gave best results
    # gamelog_roll2_df = pd.read_pickle('data/gamelog_data_2_game_rolling.pkl')
    # gamelog_roll3_df = pd.read_pickle('data/gamelog_data_3_game_rolling.pkl')
    # gamelog_roll4_df = pd.read_pickle('data/gamelog_data_4_game_rolling.pkl')
    # gamelog_roll5_df = pd.read_pickle('data/gamelog_data_5_game_rolling.pkl')
    # gamelog_roll6_df = pd.read_pickle('data/gamelog_data_6_game_rolling.pkl')
    # gamelog_roll7_df = pd.read_pickle('data/gamelog_data_7_game_rolling.pkl')

    final_stats_2017_season = pd.read_pickle('data/season2017_final_stats_5_game_rolling.pkl')
    final_stats_2018_season = pd.read_pickle('data/season2018_final_stats_5_game_rolling.pkl')
    gamelog_roll5_df = pd.read_pickle('data/gamelog_data_5_game_rolling.pkl')
    team_experience_df = pd.read_pickle('data/team_experience.pkl')
    team_clusters_df = pd.read_pickle('data/team_clusters.pkl')
    odds_df = pd.read_pickle('data/odds_data.pkl')


    '''
    Matchups for modeling from Gamelog (all rolling averages), experience and cluster data.
    '''

    # gamelog_experience_cluster_merge(gamelog_roll2_df, team_experience_df, team_clusters_df, 'game_2_roll.pkl')
    # gamelog_experience_cluster_merge(gamelog_roll3_df, team_experience_df, team_clusters_df, 'game_3_roll.pkl')
    # gamelog_experience_cluster_merge(gamelog_roll4_df, team_experience_df, team_clusters_df, 'game_4_roll.pkl')
    # gamelog_experience_cluster_merge(gamelog_roll5_df, team_experience_df, team_clusters_df, 'game_5_roll.pkl')
    # gamelog_experience_cluster_merge(gamelog_roll6_df, team_experience_df, team_clusters_df, 'game_6_roll.pkl')
    # gamelog_experience_cluster_merge(gamelog_roll7_df, team_experience_df, team_clusters_df, 'game_7_roll.pkl')

    '''5 game rolling ave gave best stats'''
    # gamelog_experience_cluster_merge(gamelog_roll5_df, team_experience_df, team_clusters_df, 'game_5_roll.pkl')
    final_stats_experience_cluster_merge(final_stats_2017_season, team_experience_df, team_clusters_df, 'final_stats_2017.pkl')
    final_stats_experience_cluster_merge(final_stats_2018_season, team_experience_df, team_clusters_df, 'final_stats_2018.pkl')
