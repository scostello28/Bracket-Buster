import pandas as pd
import pickle

'''
Read in gamelog data with all rolling average windows, team experience data, and cluster data.
'''

gamelog_roll2_df = pd.read_pickle('data/gamelog_data_2_game_rolling.pkl')
gamelog_roll3_df = pd.read_pickle('data/gamelog_data_3_game_rolling.pkl')
gamelog_roll4_df = pd.read_pickle('data/gamelog_data_4_game_rolling.pkl')
gamelog_roll5_df = pd.read_pickle('data/gamelog_data_5_game_rolling.pkl')
gamelog_roll6_df = pd.read_pickle('data/gamelog_data_6_game_rolling.pkl')
gamelog_roll7_df = pd.read_pickle('data/gamelog_data_7_game_rolling.pkl')
team_experience_df = pd.read_pickle('data/team_experience.pkl')
team_clusters_df = pd.read_pickle('data/team_clusters.pkl')

def gamelog_merge(df, window):
    '''
    INPUT: DataFrame
    OUTPUT: DataFrame with matching IDs merged to same row (1 game per row!)
    '''

    '''Add Unique ID for Merge'''
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
    OPcols = ['OP{}'.format(col) if col != 'ID' else col for col in g2cols]
    df2.columns = OPcols

    '''Merge games instance DataFrames'''
    df = pd.merge(df1, df2, how='left', on='ID')

    '''Drop redundant Opp column and any games where there is no data
    for oppenent'''
    df = df.drop(['Date', 'Ws', 'Opp', 'count', 'ID', 'count', 'matchup', 'Tm', 'OPTm'], axis=1) #'just_date',
    df = df.dropna()

    '''Save to pickle'''
    filepath = 'model_data/gamelogs_{}.pkl'.format(str(window))
    print(filepath)
    df.to_pickle(filepath)

def gamelog_experience_merge(gamelog_df, experience_df, window):
    '''
    INPUT: Gamelog DataFrame and experience DataFrame
    OUTPUT: DataFrame with matching IDs merged to same row (1 game per row!)
    '''

    '''Generate ID for merge'''
    gamelog_df = gamelog_df.apply(gamelog_ID, axis=1)
    experience_df = experience_df.apply(ID, axis=1)

    '''Drop Season columns generated with ID creation'''
    gamelog_df.drop(['Season'], axis=1, inplace=True)
    experience_df.drop(['Season', 'Team'], axis=1, inplace=True)

    '''merge experience DataFrame into gamelog DataFrame'''
    df = gamelog_df.merge(experience_df, on='ID', how='left')

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
    df = df.drop(['Date', 'Ws', 'Opp', 'count', 'ID', 'count', 'matchup', 'Tm', 'OPTm'], axis=1) #'just_date',
    df = df.dropna()

    filepath = 'model_data/gamelog_{}_exp.pkl'.format(str(window))
    print(filepath)
    df.to_pickle(filepath)

def gamelog_experience_cluster_merge(gamelog_df, experience_df, cluster_df, window):
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
    df = df.drop(['Date', 'Ws', 'Opp', 'count', 'ID', 'count', 'matchup', 'Tm', 'OPTm'], axis=1) #'just_date',
    df = df.dropna()

    filepath = 'model_data/gamelog_{}_exp_clust.pkl'.format(str(window))
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
    Matchups for modeling from Gamelog (all rolling averages) data.
    '''

    gamelog_roll2 = gamelog_merge(gamelog_roll2_df, window=2)
    gamelog_roll3 = gamelog_merge(gamelog_roll3_df, window=3)
    gamelog_roll4 = gamelog_merge(gamelog_roll4_df, window=4)
    gamelog_roll5 = gamelog_merge(gamelog_roll5_df, window=5)
    gamelog_roll6 = gamelog_merge(gamelog_roll6_df, window=6)
    gamelog_roll7 = gamelog_merge(gamelog_roll7_df, window=7)

    '''
    Matchups for modeling from Gamelog (all rolling averages) and experience data.
    '''

    gamelog_roll2_exp_matchups = gamelog_experience_merge(gamelog_roll2_df, team_experience_df, window=2)
    gamelog_roll3_exp_matchups = gamelog_experience_merge(gamelog_roll3_df, team_experience_df, window=3)
    gamelog_roll4_exp_matchups = gamelog_experience_merge(gamelog_roll4_df, team_experience_df, window=4)
    gamelog_roll5_exp_matchups = gamelog_experience_merge(gamelog_roll5_df, team_experience_df, window=5)
    gamelog_roll6_exp_matchups = gamelog_experience_merge(gamelog_roll6_df, team_experience_df, window=6)
    gamelog_roll7_exp_matchups = gamelog_experience_merge(gamelog_roll7_df, team_experience_df, window=7)

    '''
    Matchups for modeling from Gamelog (all rolling averages), experience and cluster data.
    '''

    gamelog_roll2_exp_matchups = gamelog_experience_cluster_merge(gamelog_roll2_df, team_experience_df, team_clusters_df, window=2)
    gamelog_roll3_exp_matchups = gamelog_experience_cluster_merge(gamelog_roll3_df, team_experience_df, team_clusters_df, window=3)
    gamelog_roll4_exp_matchups = gamelog_experience_cluster_merge(gamelog_roll4_df, team_experience_df, team_clusters_df, window=4)
    gamelog_roll5_exp_matchups = gamelog_experience_cluster_merge(gamelog_roll5_df, team_experience_df, team_clusters_df, window=5)
    gamelog_roll6_exp_matchups = gamelog_experience_cluster_merge(gamelog_roll6_df, team_experience_df, team_clusters_df, window=6)
    gamelog_roll7_exp_matchups = gamelog_experience_cluster_merge(gamelog_roll7_df, team_experience_df, team_clusters_df, window=7)
