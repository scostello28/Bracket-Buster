import pickle
import pandas as pd
import numpy as np
import random
from collections import Counter
from filters import pre_matchup_feature_selection
from scraping_utils import check_for_file, read_seasons
from model_utils import read_model

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def read_bracket(bracket_path):
    try:
        with open(bracket_path, 'r') as f:
            bracket = f.read()
    except FileNotFoundError:
        print(f'{bracket_path} does not exist' )
        raise 
        
    bracket = [team.strip().strip("'") for team in bracket.split('\n') if team.strip().strip("'").find('#') < 0]
    print(bracket)
    
    return bracket


class BracketGen:
    
    def __init__(self, bracket, pickled_model_path, final_stats_df, season, tcf=True):
        self.first_round = bracket
        self.second_round = None
        self.sweet16 = None
        self.elite8 = None
        self.final4 = None
        self.championship = None
        self.champion = None
        self.k = None
        self.season = season

        self.model = read_model(pickled_model_path)
        if isinstance(self.model, dict):
            self.models = self.model
            self.model = None
        else:
            self.models = None

        self.final_stats_df = final_stats_df
        self.tcf = tcf

    def merge(self, team1, team2):
        '''
        INPUT: DataFrame
        OUTPUT: DataFrame with matching IDs merged to same row
        '''
        if self.tcf:
            df = self.final_stats_df[
                [
                    'Tm', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
                    'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
                    'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2', 'G0',
                    'G1', 'G2', 'G3'
                    ]
            ]

        else:
            df = self.final_stats_df[
                [
                    'Tm', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
                    'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos'
                    ]
                ]

        '''Create separate dataframes for 1st and 2nd instances of games'''
        df1 = df.loc[df['Tm'] == team1,:]
        df2 = df.loc[df['Tm'] == team2,:]

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

        return dfout, team1, team2

    # @staticmethod
    def game_predict(self, model, matchup, matchup_reversed, team1, team2, probability, verbose=True):
        '''Predict on matchup'''
        # print(f'matchup: {matchup}')
        if verbose: print(f'{team1} vs {team2}')
        prob = model.predict_proba(matchup)
        prob_reversed = model.predict_proba(matchup_reversed)
        team1_prob = (prob[0][1] + prob_reversed[0][0]) / 2 * 100
        team2_prob = (prob[0][0] + prob_reversed[0][1]) / 2 * 100

        if probability:
            if verbose: print(f"{team1}: {team1_prob} | {team2}: {team2_prob}")
            # return BracketGen._pick_winner_probability(team1, team2, team1_prob, team2_prob)
            return self._pick_winner_probability(team1, team2, team1_prob, team2_prob)
        else:
            if team1_prob > team2_prob:
                return team1
            else:
                return team2

    @staticmethod
    def game_predict_ave(model, matchup, matchup_reversed, team1, team2, verbose=True):
        '''Predict on matchup'''
        # print(f'matchup: {matchup}')
        if verbose: print(f'{team1} vs {team2}')
        prob = model.predict_proba(matchup)
        prob_reversed = model.predict_proba(matchup_reversed)
        team1_prob = (prob[0][1] + prob_reversed[0][0]) / 2 * 100
        team2_prob = (prob[0][0] + prob_reversed[0][1]) / 2 * 100

        return team1_prob, team2_prob

    @staticmethod
    def _pick_winner_random(team1, team2):
        matchup = [team1, team2]
        winner = matchup[randint(0,1)]
        return winner
    
    def _pick_winner(self, team1, team2, probability=False):
        matchup, team1, team2 = self.merge(team1, team2)
        matchup_reversed, team2_rev, team1_rev = self.merge(team2, team1)
        return self.game_predict(self.model, matchup, matchup_reversed, team1, team2, probability)
        # return BracketGen.game_predict(self.model, matchup, matchup_reversed, team1, team2, probability)

    def _pick_winner_ave(self, team1, team2):
        matchup, team1, team2 = self.merge(team1, team2)
        matchup_reversed, team2_rev, team1_rev = self.merge(team2, team1)

        team1_prob = 0
        team2_prob = 0
        for model, weight in self.models.items():
            team1_prob_i, team2_prob_i = BracketGen.game_predict_ave(model, matchup, matchup_reversed, team1, team2, verbose=False)
            team1_prob += team1_prob_i * weight
            team2_prob += team2_prob_i * weight
        
        if team1_prob > team2_prob:
            return team1
        else:
            return team2

    # @staticmethod
    def _pick_winner_probability(self, team_1, team_2, prob_1, prob_2):
        """
        Selects between two teams based on their probability of winning.
        """
        options = [team_1, team_2]
        weights = [prob_1, prob_2]
        
        selection = random.choices(options, weights=weights, k=self.k)
        counts = Counter(selection)
        most_common_item, frequency = counts.most_common(1)[0]
        return most_common_item
    
    def _pick_round(self, round_list, probability=False):
        next_round = []
        i = 0
        while i <= len(round_list)-2:
            team1, team2 = round_list[i], round_list[i+1]
            winner = self._pick_winner(team1, team2, probability)
            next_round.append(winner)
            i += 2
        return next_round

    def _pick_round_ave(self, round_list):
        next_round = []
        i = 0
        while i <= len(round_list)-2:
            team1, team2 = round_list[i], round_list[i+1]
            winner = self._pick_winner_ave(team1, team2)
            next_round.append(winner)
            i += 2
        return next_round
    
    def gen_bracket(self, verbose=True, bracket_name=None, model_ave=False, brackets_dir=None, k=None, probability=False):

        self.k = k

        if model_ave:
            self.second_round = self._pick_round_ave(self.first_round)
            self.sweet16 = self._pick_round_ave(self.second_round)
            self.elite8 = self._pick_round_ave(self.sweet16)
            self.final4 = self._pick_round_ave(self.elite8)
            self.championship = self._pick_round_ave(self.final4)
            self.champion = self._pick_round_ave(self.championship)
        else:
            self.second_round = self._pick_round(self.first_round, probability)
            self.sweet16 = self._pick_round(self.second_round, probability)
            self.elite8 = self._pick_round(self.sweet16, probability)
            self.final4 = self._pick_round(self.elite8, probability)
            self.championship = self._pick_round(self.final4, probability)
            self.champion = self._pick_round(self.championship, probability)

        if bracket_name:

            if probability:
                bracket_file_name = f"{bracket_name}_prob_k{self.k}_{random.randint(0, 9999)}_{self.champion[0]}_{self.season}.txt"
            else:
                bracket_file_name = f"{bracket_name}_{self.champion[0]}_{season}.txt"

            f = open(f"{brackets_dir}/{bracket_file_name}", 'w')
            print("First Round", file=f)
            print("-----------", file=f)
            BracketGen.print_list(self.first_round, f)
            print("\n", file=f)
            print("Second Round", file=f)
            print("------------", file=f)
            BracketGen.print_list(self.second_round, f)
            print("\n", file=f)
            print("Sweet 16", file=f)
            print("-------", file=f)
            BracketGen.print_list(self.sweet16, f)
            print("\n", file=f)
            print("Elite 8", file=f)
            print("-------", file=f)
            BracketGen.print_list(self.elite8, f)
            print("\n", file=f)
            print("Final 4", file=f)
            print("------", file=f)
            BracketGen.print_list(self.final4, f)
            print("\n", file=f)
            print("Championship", file=f)
            print("------------", file=f)
            BracketGen.print_list(self.championship, f)
            print("\n", file=f)
            print("Champion", file=f)
            print("--------", file=f)
            BracketGen.print_list(self.champion, f)
            f.close()
    
        if verbose:
            print(f"First_round: {self.first_round}")
            print("\n")
            print(f"Second_round: {self.second_round}")
            print("\n")
            print(f"Sweet16: {self.sweet16}")
            print("\n")
            print(f"Elite8: {self.elite8}")
            print("\n")
            print(f"Final4: {self.final4}")
            print("\n")
            print(f"Championship: {self.championship}")
            print("\n")
            print(f"Champion: {self.champion}")

    @staticmethod
    def print_list(l, f):
        i=0
        while i < len(l):
            print(" v ".join(l[i: i+2]), file=f)
            i+=2


if __name__ == '__main__':

    season = read_seasons(seasons_path='seasons_list.txt')[-1]
    root_dir = "/Users/sean/Documents/bracket_buster"
    brackets_dir = "repo/brackets"
    print(season)

    bracket = read_bracket(bracket_path=f"{root_dir}/{brackets_dir}/{season}/initial_bracket_{season}.txt")

    models = {
        "lr_tcf": f"{root_dir}/fit_models/{season}/lr_tcf_{season}_fit_model.joblib",
        "rf_tcf": f"{root_dir}/fit_models/{season}/rf_tcf_{season}_fit_model.joblib",
        "gb_tcf": f"{root_dir}/fit_models/{season}/gb_tcf_{season}_fit_model.joblib",
        "lr": f"{root_dir}/fit_models/{season}/lr_{season}_fit_model.joblib",
        "rf": f"{root_dir}/fit_models/{season}/rf_{season}_fit_model.joblib",
        "gb": f"{root_dir}/fit_models/{season}/gb_{season}_fit_model.joblib"
    }  

    final_stats_df = pd.read_pickle(f'{root_dir}/data/3_model_data/{season}/season{season}_final_stats.pkl')
    finalgames_data = final_stats_df[final_stats_df['GameType'] == f'season{season}']
    finalgames_exp_tcf = pre_matchup_feature_selection(finalgames_data, 'exp_tcf')
    finalgames = pre_matchup_feature_selection(finalgames_data, 'gamelogs')

    gb_tcf = BracketGen(
        bracket=bracket, 
        pickled_model_path=models["gb_tcf"], 
        final_stats_df=finalgames_exp_tcf, 
        season=season,
        tcf=True
        )
    gb_tcf.gen_bracket(
        bracket_name=f"gb_tcf",
        brackets_dir=f"{root_dir}/{brackets_dir}/{season}",
        probability=True,
        k=99999
        )

    # lr_tcf = BracketGen(
    #     bracket=bracket, 
    #     pickled_model_path=models["lr_tcf"], 
    #     final_stats_df=finalgames_exp_tcf, 
    #     tcf=True)
    # lr_tcf.gen_bracket(
    #     bracket_name=f"lr_tcf",
    #     brackets_dir=f"{root_dir}/{brackets_dir}/{season}"
    #     )

    # rf_tcf = BracketGen(
    #     bracket=bracket, 
    #     pickled_model_path=models["rf_tcf"], 
    #     final_stats_df=finalgames_exp_tcf, 
    #     tcf=True)
    # rf_tcf.gen_bracket(
    #     bracket_name=f"rf_tcf",
    #     brackets_dir=f"{root_dir}/{brackets_dir}/{season}"
    #     )

    # gb_tcf = BracketGen(
    #     bracket=bracket, 
    #     pickled_model_path=models["gb_tcf"], 
    #     final_stats_df=finalgames_exp_tcf, 
    #     tcf=True)
    # gb_tcf.gen_bracket(
    #     bracket_name=f"gb_tcf",
    #     brackets_dir=f"{root_dir}/{brackets_dir}/{season}"
    #     )

    # lr = BracketGen(
    #     bracket=bracket, 
    #     pickled_model_path=models["lr"], 
    #     final_stats_df=finalgames, 
    #     tcf=False)
    # lr.gen_bracket(
    #     bracket_name=f"lr",
    #     brackets_dir=f"{root_dir}/{brackets_dir}/{season}"
    #     )

    # rf = BracketGen(
    #     bracket=bracket, 
    #     pickled_model_path=models["rf"], 
    #     final_stats_df=finalgames, 
    #     tcf=False)
    # rf.gen_bracket(
    #     bracket_name=f"rf",
    #     brackets_dir=f"{root_dir}/{brackets_dir}/{season}"
    #     )

    # gb = BracketGen(
    #     bracket=bracket, 
    #     pickled_model_path=models["gb"], 
    #     final_stats_df=finalgames, 
    #     tcf=False)
    # gb.gen_bracket(
    #     bracket_name=f"gb",
    #     brackets_dir=f"{root_dir}/{brackets_dir}/{season}"
    #     )

    # ave_models = {
    #     models["gb"]: .95,
    #     # models["rf"]: .25,
    #     models["lr"]: .05,
    # }

    # ave = BracketGen(
    #     bracket=bracket, 
    #     pickled_model_path=ave_models, 
    #     final_stats_df=finalgames_exp_tcf, 
    #     tcf=True)
    # ave.gen_bracket(
    #     bracket_name=f"model_ave", 
    #     model_ave=True,
    #     brackets_dir=f"{root_dir}/{brackets_dir}/{season}"
    #     )
