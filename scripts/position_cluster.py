import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pdb

from scraping_utils import check_for_file, read_seasons


'''Functions'''
def guards(df):
    df = df[df['Pos'] == 'G']
    return df

def forwards(df):
    df = df[df['Pos'] == 'F']
    return df

def centers(df):
    df = df[df['Pos'] == 'C']
    return df

def ID(row):
    row['ID'] = ",".join([row['Team'], str(row['Season'])])
    return row

def pos_dfs(df):
    '''
    Create DataFrames for each posiiton with select features
    INPUT: Player stats Dataframe
    OUTPUT: Dataframes for each position ready forclustering
    '''
    # Features to Cluster Centers
    c_reduced_df_cols = ['Player', 'MP', '2P', '3P', 'TRB', 'AST', 'STL',
                         'BLK', 'TOV', 'PTS', 'Team', 'Season', 'Pos', 'Height']

    c_df_r = df[c_reduced_df_cols]

    # Features to Cluster Forwards
    f_reduced_df_cols = ['Player', 'MP', '2P', '2PA', '3P', '3PA', 'TRB', 'AST', 'STL',
                         'BLK', 'TOV', 'PTS', 'Team', 'Season', 'Pos']

    f_df_r = df[f_reduced_df_cols]

    # Features to Cluster Guards
    g_reduced_df_cols = ['Player', 'MP', '3P', 'AST', 'STL', 'TOV',
                         'PTS', 'TRB', 'Team', 'Season', 'Pos']

    g_df_r = df[g_reduced_df_cols]

    # Drop NaNs from reduced DataFrames
    c_df_r = centers(c_df_r.dropna())
    f_df_r = forwards(f_df_r.dropna())
    g_df_r = guards(g_df_r.dropna())

    return c_df_r, f_df_r, g_df_r

def vectorize_and_standardize(df):
    '''
    INPUT: DataFrame
    OUTPUT: Standardized NumPy Matrix, and arrays for players and positions
    '''
    # Vectorize
    player = df['Player'].to_numpy()
    position = df['Pos'].to_numpy()
    X = df.drop(['Player', 'MP', 'Team', 'Season', 'Pos'], axis=1).to_numpy()

    # Standardize
    scale = StandardScaler()
    X = scale.fit_transform(X)

    return X, player, position

def create_clusters(X, nclusters):
    '''
    INPUT: X Matrix and number of Clusters
    OUTPUT: Cluster labels for each observation
    '''
    kmeans = KMeans(n_clusters=nclusters, init='k-means++', n_init=20, max_iter=500, tol=0.0001,
                        verbose=0, random_state=None,
                        copy_x=True, algorithm='auto')
    kmeans.fit(X)
    return kmeans.labels_

def add_clusters_to_dfs(df, clusters):
    df['Cluster'] = clusters
    return df

def position_cluster(row):
    '''
    INPUT: row in DataFrame
    OUPTUT: concatenation of position and cluster number
    '''
    row['pos_cluster'] = row['Pos'] + str(row['Cluster'])
    return row

def concatinate_dataframes(df1, df2, df3):
    '''
    Concatenate position specific DataFrames into one with cluster position column
    INPUT: Three DataFrames (one for each position)
    OUTPUT: One Concatenated DataFrame
    '''
    pos1 = df1[['Player', 'MP', 'Team', 'Season', 'Cluster', 'Pos']]
    pos2 = df2[['Player', 'MP', 'Team', 'Season', 'Cluster', 'Pos']]
    pos3 = df3[['Player', 'MP', 'Team', 'Season', 'Cluster', 'Pos']]
    positions = [pos1, pos2, pos3]
    players = pd.concat(positions)
    players = players.apply(position_cluster, axis=1)
    return players

def team_and_season_mp_by_cluster(df):
    '''
    INPUT: DataFrame of each player and assigned Cluster
    OUTPUT: DataFrame of each team for each season with percentage of munutes played by each cluster
    '''
    pivot_df = pd.pivot_table(df, values='MP', index=['Team', 'Season'],
                         columns=['pos_cluster'], aggfunc='sum', fill_value=0)
    pivot_df = pivot_df.reset_index()

    team_and_season = pivot_df.iloc[:, :2].to_numpy()
    clusts = pivot_df.iloc[:, 2:].to_numpy()
    cols = pivot_df.columns.tolist()
    mp = clusts.sum(axis=1)
    clustsnorm = clusts / mp.reshape(-1, 1)
    clusters_df = pd.DataFrame(np.hstack((team_and_season, clustsnorm)), columns=cols)
    return clusters_df

def team_and_season_mp_by_class(df):
    '''
    INPUT: DataFrame of each player and Class
    OUTPUT: DataFrame of each team for each season with percentage of munutes played by each class
    '''
    class_df = df[['Team', 'Season', 'Class', 'MP']]

    pivot_df = pd.pivot_table(df, values='MP', index=['Team', 'Season'],
                         columns=['Class'], aggfunc='sum', fill_value=0)
    pivot_df = pivot_df.reset_index()

    team_and_season = pivot_df.iloc[:, :2].to_numpy()
    classes = pivot_df.iloc[:, 2:].to_numpy()
    cols = pivot_df.columns.tolist()
    mp = classes.sum(axis=1)
    classesnorm = classes / mp.reshape(-1, 1)
    cl = np.array(cols[2:])
    cols = cols[:2]
    cols.append('exp_factor')
    exp = cl * classesnorm
    expfactor = exp.sum(axis=1)
    classes_df = pd.DataFrame(np.hstack((team_and_season, expfactor.reshape(-1, 1))), columns=cols)
    return classes_df

def team_experience_level(df):
    '''
    INPUT: Player Stats DataFrame
    OUTPUT: DataFrame with mean Class
    '''
    df = df[['Team', 'Season', 'Class']].groupby(['Team', 'Season']).mean().round(2)
    df.columns = ['Team', 'Season', 'Experience']
    df = df.reset_index()
    return df

def merge_dfs(df1, df2):
    '''
    INPUT: Two DataFrames with Team and Season columns
    OUTPUT: One DataFrame that is the combination of the two
    '''
    df1 = df1.apply(ID, axis=1)
    df2 = df2.apply(ID, axis=1)
    df2 = df2.drop(['Team', 'Season'], axis=1)
    merged_df = df1.merge(df2, on='ID', how='left')
    merged_df = merged_df.drop(['ID'], axis=1)
    return merged_df


if __name__ == '__main__':

    source_dir = "2_full_season_data"
    output_dir = "2_full_season_data"

    season = read_seasons(seasons_path='seasons_list.txt')[-1]
    df = pd.read_pickle(f"{source_dir}/player_stats_full-{season}.pkl")

    # Create DataFrames by position ready for clustering
    centers_df, forwards_df, guards_df = pos_dfs(df)

    # Vecotrize and Standardize DataFrames
    X_c, player_c, position_c = vectorize_and_standardize(centers_df)
    X_f, player_f, position_f = vectorize_and_standardize(forwards_df)
    X_g, player_g, position_g = vectorize_and_standardize(guards_df)

    # Create Clusters
    center_clusters = create_clusters(X_c, nclusters=3)
    forward_clusters = create_clusters(X_f, nclusters=3)
    guard_clusters = create_clusters(X_g, nclusters=4)

    # Add Cluster column to dataframe
    centers_df = add_clusters_to_dfs(centers_df, center_clusters)
    forwards_df = add_clusters_to_dfs(forwards_df, forward_clusters)
    guards_df = add_clusters_to_dfs(guards_df, guard_clusters)

    # Create one Dataframe to rule them all
    players_df = concatinate_dataframes(centers_df, forwards_df, guards_df)

    # Sum Position cluster minutes played by Team and Season
    team_clusters_df = team_and_season_mp_by_cluster(players_df)
    print(f"{output_dir}/team_clusters-{season}.pkl")
    team_clusters_df.to_pickle(f"{output_dir}/team_clusters-{season}.pkl")

    # Create Team Experience Level DataFrame
    team_experience_df = team_and_season_mp_by_class(df)
    print(f"{output_dir}/team_experience-{season}.pkl")
    team_experience_df.to_pickle(f"{output_dir}/team_experience-{season}.pkl")
