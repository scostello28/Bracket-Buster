import joblib
import pickle
import pandas as pd
import numpy as np
from filters import pre_matchup_feature_selection
from scraping_utils import read_seasons
from model_utils import read_model

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def merge(df, team1, team2, tcf):
    '''
    INPUT: DataFrame
    OUTPUT: DataFrame with matching IDs merged to same row
    '''
    if tcf:
        df = df[['Tm', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
                 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
                 'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2', 'G0',
                 'G1', 'G2', 'G3']]

    else:
        df = df[['Tm', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
                 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos']]

    '''Create separate dataframes for 1st and 2nd instances of games'''
    df1 = df.loc[df['Tm'] == team1,:]
    df2 = df.loc[df['Tm'] == team2,:]

    '''Select needed columns from 2nd instance DataFrame and
    rename te prepare for pending left merge'''
    # df2_stats = df2  #.iloc[:, 5:19]

    df2cols = df2.columns.tolist()
    OPcols = ['OP{}'.format(col) for col in df2cols]
    df2.columns = OPcols

    '''Force unique ID to merge on'''
    df1['game'] = 'game'
    df2['game'] = 'game'

    '''Merge games instance DataFrames'''
    dfout = pd.merge(df1, df2, how='left', on='game')

    '''Drop uneeded columns'''
    dfout = dfout.drop(['game', 'Tm', 'OPTm'], axis=1)

    return dfout

def game_predict(model, matchup, matchup_reversed, team1, team2):

    '''Predict on matchup'''
    prob = model.predict_proba(matchup)
    prob_reversed = model.predict_proba(matchup_reversed)
    team1_prob = (prob[0][1] + prob_reversed[0][0]) / 2 * 100
    team2_prob = (prob[0][0] + prob_reversed[0][1]) / 2 * 100

    '''Print results'''
    if team1_prob < team2_prob:
        print('{} wins and {} loses!'.format(team2, team1))
        print('{} has {}% chance to win.'.format(team1, int(round(team1_prob))))
        print('{} has {}% chance to win.'.format(team2, int(round(team2_prob))))
    else:
        print('{} wins and {} loses!'.format(team1, team2))
        print('{} has {}% chance to win.'.format(team1, int(round(team1_prob))))
        print('{} has {}% chance to win.'.format(team2, int(round(team2_prob))))

def make_prediction(fit_model_path, final_games, team1, team2, tcf=True):
    
    model = read_model(fit_model_path)
    matchup = merge(final_games, team1, team2, tcf)
    matchup_reversed = merge(final_games, team2, team1, tcf)
    game_predict(model, matchup, matchup_reversed, team1, team2)

def make_prediction_old(pickled_model, pickled_model_exp_tcf, final_games, finalgames_exp_tcf, team1, team2):
    with open(pickled_model, 'rb') as f:
        model = pickle.load(f)

    with open(pickled_model_exp_tcf, 'rb') as f_exp_tcf:
        model_exp_tcf = pickle.load(f_exp_tcf)

    # team1 = str(input('team1: '))
    # team2 = str(input('team2: '))

    print('\n')
    print('w/o TCF')
    matchup = merge(finalgames, team1, team2, tcf=False)
    matchup_reversed = merge(finalgames, team2, team1, tcf=False)
    game_predict(model, matchup, matchup_reversed, team1, team2)
    print('\n')
    print('w/ TCF')
    matchup_exp_tcf = merge(finalgames_exp_tcf, team1, team2, tcf=True)
    matchup_exp_tcf_reversed = merge(finalgames_exp_tcf, team2, team1, tcf=True)
    game_predict(model_exp_tcf, matchup_exp_tcf, matchup_exp_tcf_reversed, team1, team2)


def game_predict_ave(model, matchup, matchup_reversed, team1, team2):

    '''Predict on matchup'''
    prob = model.predict_proba(matchup)
    prob_reversed = model.predict_proba(matchup_reversed)
    team1_prob = (prob[0][1] + prob_reversed[0][0]) / 2 * 100
    team2_prob = (prob[0][0] + prob_reversed[0][1]) / 2 * 100

    return team1_prob, team2_prob

def make_prediction_ave(models, final_games, team1, team2, tcf=True):
    
    matchup = merge(final_games, team1, team2, tcf=tcf)
    matchup_reversed = merge(final_games, team2, team1, tcf=tcf)
    
    team1_prob = 0
    team2_prob = 0
    for model, weight in models.items():
        team1_prob_i, team2_prob_i = game_predict_ave(model, matchup, matchup_reversed, team1, team2)
        team1_prob += team1_prob_i * weight
        team2_prob += team2_prob_i * weight

    '''Print results'''
    if team1_prob < team2_prob:
        print('{} wins and {} loses!'.format(team2, team1))
        print('{} has {}% chance to win.'.format(team1, int(round(team1_prob))))
        print('{} has {}% chance to win.'.format(team2, int(round(team2_prob))))
    else:
        print('{} wins and {} loses!'.format(team1, team2))
        print('{} has {}% chance to win.'.format(team1, int(round(team1_prob))))
        print('{} has {}% chance to win.'.format(team2, int(round(team2_prob))))

if __name__ == '__main__':

    season = read_seasons(seasons_path='seasons_list.txt')[-1]

    root_dir = "/Users/sean/Documents/bracket_buster"
    brackets_dir = "repo/brackets"
    model_dir = "fit_models"

    final_stats_df = pd.read_pickle(f'{root_dir}/data/3_model_data/{season}/season{season}_final_stats.pkl')
    finalgames_data = final_stats_df[final_stats_df['GameType'] == f'season{season}']
    finalgames_exp_tcf = pre_matchup_feature_selection(finalgames_data, 'exp_tcf')
    finalgames = pre_matchup_feature_selection(finalgames_data, 'gamelogs')

    lr_model_path = f'lr_{season}_fit_model.joblib'
    rf_model_path = f'rf_{season}_fit_model.joblib'
    gb_model_path = f'gb_{season}_fit_model.joblib'

    lr_model_exp_tcf_path = f'lr_tcf_{season}_fit_model.joblib'
    rf_model_exp_tcf_path = f'rf_tcf_{season}_fit_model.joblib'
    gb_model_exp_tcf_path = f'gb_tcf_{season}_fit_model.joblib'

    # pickled_model = gb_model
    model = gb_model_exp_tcf_path

    team1 = str(input('team1: '))
    team2 = str(input('team2: '))

    make_prediction(
        fit_model_path=f"{root_dir}/{model_dir}/{season}/{gb_model_exp_tcf_path}", 
        final_games=finalgames_exp_tcf, 
        team1=team1, 
        team2=team2, 
        tcf=True
        )

    ####################
    # models = {
    #     'lr': (lr_model, lr_model_exp_tcf), 
    #     'rf_model': (rf_model, rf_model_exp_tcf), 
    #     'gb_model': (gb_model, gb_model_exp_tcf)
    # }

    # for model_name, model in models.items():
    #     print('\n')
    #     print(model_name)
    #     make_prediction(model[0], model[1], finalgames, finalgames_exp_tcf, team1, team2)

    ####################
    # with open(pickled_model, 'rb') as f:
    #     model = pickle.load(f)

    # with open(pickled_model_exp_tcf, 'rb') as f_exp_tcf:
    #     model_exp_tcf = pickle.load(f_exp_tcf)

    # team1 = str(input('team1: '))
    # team2 = str(input('team2: '))

    # print('\n')
    # print('w/o TCF')
    # matchup = merge(finalgames, team1, team2, tcf=False)
    # matchup_reversed = merge(finalgames, team2, team1, tcf=False)
    # game_predict(model, matchup, matchup_reversed, team1, team2)
    # print('\n')
    # print('w/ TCF')
    # matchup_exp_tcf = merge(finalgames_exp_tcf, team1, team2, tcf=True)
    # matchup_exp_tcf_reversed = merge(finalgames_exp_tcf, team2, team1, tcf=True)
    # game_predict(model_exp_tcf, matchup_exp_tcf, matchup_exp_tcf_reversed, team1, team2)

    ####################

    # print("\n")
    # print("Model AVG")
    # ave_model_path_weight_dict = {
    #     gb_model_exp_tcf: .9,
    #     # rf_model_exp_tcf: .25,
    #     lr_model_exp_tcf: .1
    # }

    # def read_models(model_path_weight_dict):
    #     models = {}
    #     for model_path, weight in model_path_weight_dict.items():
    #         with open(model_path, 'rb') as f:
    #             model = pickle.load(f)
    #         models[model] = weight
    #     return models

    # models = read_models(ave_model_path_weight_dict)

    # make_prediction_ave(models, finalgames_exp_tcf, team1, team2, tcf=True)


