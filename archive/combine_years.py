import pandas as pd
import pickle

game_logs = ['game_data/games_2014.pkl', 'game_data/games_2014.pkl', 'game_data/games_2015.pkl', 'game_data/games_2016.pkl', 'game_data/games_2017.pkl']

def combine_years(game_logs):
    games_all_years = pd.DataFrame()
    for game_log in game_logs:
        games = pd.read_pickle(game_log)
        games_all_years = games_all_years.append(games, ignore_index=True)
    games_all_years.to_pickle('games_five_years.pkl')

if __name__ == '__main__':
    combine_years(game_logs)
