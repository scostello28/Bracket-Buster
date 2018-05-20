import pandas as pd
import pickle
import pdb



def gamelogs_odds_merge_id(row):
    '''
    ID generator to merge odds data with gamelog data for gamelog dataframe
    '''
    row['ID'] = '{},{}'.format(",".join(sorted([row['Tm'], row['OPTm']])), row['GameType'][-4:] + row['Date'][-6:])
    return row

def odds_merge_id(row):
    '''
    ID generator to merge odds data with gamelog data for odds dataframe
    '''
    row['ID'] = '{},{}'.format(",".join(sorted([row['Team'], row['Team_v']])), row['Date'])
    return row

def final_p(row):
    if row['Tm'] == row['Team']:
        row['final_p'] = row['p']
    elif row['Tm'] == row['Team_v']:
        row['final_p'] = row['p_v']
    return row

def merge(gamelog_df, odds_df):
    gamelog_df = gamelog_df.apply(gamelogs_odds_merge_id, axis=1)
    odds_df = odds_df.apply(odds_merge_id, axis=1)
    odds_df.drop(['Date'], axis=1, inplace=True)
    merged_df = pd.merge(gamelog_df, odds_df, how='left', on='ID')
    merged_df.dropna(inplace=True)
    merged_df = merged_df.apply(final_p, axis=1)
    final_filepath = 'final_model_data/gamelog_exp_clust_odds.pkl'
    print(final_filepath)
    merged_df.to_pickle(final_filepath)

if __name__ == '__main__':
    gamelog_data = pd.read_pickle('model_data/gamelog_5_exp_clust.pkl')
    odds_data = pd.read_pickle('data/odds_data.pkl')

    merge(gamelog_data, odds_data)
