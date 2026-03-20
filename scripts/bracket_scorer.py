import os
import pandas as pd

from scraping_utils import read_seasons


def parse_bracket(file_path):
    
    bracket_data = {
        'First Round': [],
        'Second Round': [],
        'Sweet 16': [],
        'Elite 8': [],
        'Final 4': [],
        'Championship': [],
        'Champion': []
    }
    current_round = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Skip empty lines and header underscores (---)
            if not line or line.startswith('-'):
                continue

            # Check if the line is a matchup or a round title
            
            if line.strip() in bracket_data.keys():
                current_round = line.strip()
            elif ' v ' in line:
                teams = tuple(line.split(' v '))
                bracket_data[current_round].append(teams)
            elif current_round == 'Champion':
                bracket_data[current_round].append(line)
                
    return bracket_data


def score_brackets(actual_bracket, score_bracket):
    
    round_scores = {
        'First Round': 0,
        'Second Round': 0,
        'Sweet 16': 0,
        'Elite 8': 0,
        'Final 4': 0,
        'Championship': 0,
        'Final Score': 0
    }

    round_points = {
        'First Round': 1,
        'Second Round': 2,
        'Sweet 16': 4,
        'Elite 8': 8,
        'Final 4': 16,
        'Championship': 32
    }

    bracket_rounds = [
        'First Round', 'Second Round', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship', 'Champion'
    ]

    for round_ in bracket_rounds[1:]:

        if round_ == 'Champion':
            actual_teams = set([actual_bracket.get(round_)[0]])
            score_teams = set([score_bracket.get(round_)[0]])

        else:
            # gets the teams out of the list of tuples and into a flattened set
            actual_teams = set(team for matchup in actual_bracket.get(round_) for team in matchup)
            score_teams = set(team for matchup in score_bracket.get(round_) for team in matchup)
            
        prev_round = bracket_rounds[bracket_rounds.index(round_) - 1]
        
        correct_teams = actual_teams.intersection(score_teams)
        round_scores[prev_round] = len(correct_teams) * round_points[prev_round]

    round_scores["Final Score"] = sum([v for v in round_scores.values()])

    return round_scores


def create_bracket_score_df(season_bracket_scores, season):
    """
    Convert a siven disctionary of bracket scores and the season in to a pandas dataframe
    """
    unpacked = [(k, v) for k, v in season_bracket_scores.items()]
    model_names = [i[0] for i in unpacked]
    score_data = [i[1] for i in unpacked]
    score_df = pd.DataFrame(score_data, index=model_names).reset_index()
    score_df = score_df.rename(columns={"index": "Model"})
    score_df['Season'] = season
    
    return score_df[['Model', 'Season', 'Final Score', 'First Round', 'Second Round', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship']]


def clean_up_bracket_scores_csv(brackets_root_dir, bracket_score_dtypes):
    print("Clean up bracket_scores.csv")
    try:
        bracket_scores_df = pd.read_csv(f"{brackets_root_dir}/bracket_scores.csv", dtype=bracket_score_dtypes)
    except FileNotFoundError:
        print("bracket_scores.csv does not exist")

    bracket_scores_df = bracket_scores_df.drop_duplicates(
        keep='last', ignore_index=True
        )
    bracket_scores_df = bracket_scores_df.sort_values(by=['Season', 'Model'], ascending=False)
    bracket_scores_df.to_csv(f"{brackets_root_dir}/bracket_scores.csv", index=False)


if __name__ == '__main__':

    root_dir = "/Users/sean/Documents/bracket_buster"
    brackets_dir = "repo/brackets"
    brackets_root_dir = f"{root_dir}/{brackets_dir}"

    bracket_score_dtypes={
        'Model': str, 'Season': int, 'Final Score': int, 'First Round': int, 
        'Second Round': int, 'Sweet 16': int, 'Elite 8': int, 'Final 4': int, 'Championship': int
    }

    # check to see if bracket scores df exists
    bracket_scores_file_exists = False
    try:
        bracket_scores_df = pd.read_csv(f"{brackets_root_dir}/bracket_scores.csv", dtype=bracket_score_dtypes)
        bracket_scores_file_exists = True
    except FileNotFoundError:
        print("bracket_scores.csv does not exist")

    seasons = read_seasons(seasons_path='seasons_list.txt')   
    season_dirs = [i for i in os.listdir(brackets_root_dir) if i in [str(season) for season in seasons]]

    for season in season_dirs:

        # get list of files in season directory
        season_dir_list = []
        for file in os.listdir(f"{brackets_root_dir}/{season}"):
            if not 'initial_bracket' in file and file != '.DS_Store':
                season_dir_list.append(file)

        # only try and score brackets if the actual bracket is present
        if any('actual_bracket' in file for file in season_dir_list):
            print(f"Scoring {season} brackets")

            if bracket_scores_file_exists:
                bracket_scores_df = pd.read_csv(f"{brackets_root_dir}/bracket_scores.csv", dtype=bracket_score_dtypes)
                print("Read bracket_scores.csv")
            
            # collect season brackets
            season_brackets = {}
            for file in season_dir_list:
                model_name = file[:file.find(f'_{season}.txt')]

                # only score bracket if it hasn't already been scored and logged
                if bracket_scores_file_exists:
                    print(f"parsing: {season}/{file}")
                    if bracket_scores_df[(bracket_scores_df['Model'] == model_name) & (bracket_scores_df['Season'] == int(season))].shape[0] == 0:
                        season_brackets[model_name] = parse_bracket(f"{brackets_root_dir}/{season}/{file}")
                    else:
                        print(f"Model: {model_name} from Season: {season} already exists")
                else:
                    print(f"parsing: {season}/{file}")
                    season_brackets[model_name] = parse_bracket(f"{brackets_root_dir}/{season}/{file}")

            # score season brackets
            season_bracket_scores = {}
            if len(season_brackets) > 1: # actual bracket will get added
                for bracket_name, score_bracket in season_brackets.items():
                    if 'actual_bracket' not in bracket_name:
                        season_bracket_scores[bracket_name] = score_brackets(season_brackets['actual_bracket'], score_bracket=score_bracket)

                # convert season_bracket_scores into dataframe
                score_df = create_bracket_score_df(season_bracket_scores, season)
            
                # record season bracket scores
                if bracket_scores_file_exists == False:
                    print("Create bracket_scores.csv")
                    score_df.to_csv(f"{brackets_root_dir}/bracket_scores.csv", index=False)
                    bracket_scores_file_exists = True

                else:
                    score_df.to_csv(f"{brackets_root_dir}/bracket_scores.csv", mode='a', index=False, header=False)


clean_up_bracket_scores_csv(brackets_root_dir, bracket_score_dtypes)
