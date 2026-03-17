# Bracket Buster

## Pipeline

**Scrape Data**

1. sos_list_scraper.py
    - scraped strength of schedule data
2. gamelog_scraper.py
    - scrapes teams gamelog data
3. player_scraper.py
    - scrapes roster and playter per 100 posession data

**Transform Data**

4. gamelog_stats_transform.py
5. data_merger.py
    - concats seasons
    - merges roster and player_per100 data
6. position_cluster.py
    - creates position clusters
    - creates team experiecnce factor
7. matchup_creator.py
    - merges gamelogs, clustering and experience dataframes
    - slices up dataframe to create matchups
    - saves final_model_data

**Modelling**
- model_optimization.py - skipping this step for now
- model_dumper.py
- model_tests.py

**Predicting**
- win_or_lose.py
- bracket_generator.py

**Utils**
- scraping_utils.py
- filters.py

**TODO for next update**
- Add config so only need to update season in one place
- add season to file names on files that need archived
    - player_per100_full_date.pkl
    - player_stats_full.pkl
    - roster_full_data.csv
    - season-full_game_log_stats_data.pkl
    - team_clusters.pkl
    - team_experience.pkl
    - exp_gamelog_clust
- add full update shell script
    - archive past year files
    - run all scripts in order

**TODO**
- Save final brackets from each tourney
- vectorize height data in data merger player_roster_merger func
- fill missing height data in rosters instead of dropping all wiht NaNs 
- create archiving script to move all yearly data files to respective archive folders
- Create a testing framework to see which models are best
    - for full bracket 
    - for each round
        - ie. is tcf better in early rounds?
        - Can I create some ensemble of these models for better performace?

**Annual Update Prcess**
1. **Test all scraping scripts early in case they changed formatting on website**
2. Add new season to `seasons_list.txt`
3. `sos_list_scraper.py`
    - creates sos_list{season}.csv to 0_scraped_data dir
4. `gamelog_scraper.py`
    - update `add_game_type` func
        - add new year season/tourney start and end dates
        - update if else section with new conditions
    - saves `season_{season}_gamelog_data.pkl` to `0_scraped_data` dir
5. `player_scraper.py`
    - saves `player_per100_{season}_data.pkl` & `roster_{season}_data.csv` to `0_scraped_data` dir
6. `gamelog_stats_transform.py`
    - saves `season_{season}_gamelog_stats_data.pkl` agnd `season_{season}_gamelog_final_stats_data.pkl` to `1_transformed_data` dir
7. `data_merger.py`
    - Manual: archive data to year specific folder
    - saves files to `2_full_season_data` dir
        - `player_per100_full-{season}_data.pkl`
        - `roster_full-{season}_data.csv`
        - `season_full-{season}_gamelog_stats_data.pkl`
        - `player_stats_full-{season}.pkl`
8. `position_cluster.py`
    - saves files to `2_full_season_data` dir
        - `team_clusters-{season}.pkl`
        - `team_experience-{season}.pkl`
9. `matchup_creator.py`
    - Manual: archive data to year specific folder
    - creates: `gamelog_exp_clust-{season}.pkl` & `season{season}_final_stats.pkl` in `3_model_data` dir
10. `model_dumper.py`
    - Manual: archive fit models to year specific folder
    - trains logistic regression, random forest and gradient boosting models for testing and prediction
        - testing models are trained on all data up to the current season and tested on the current season to assess model hyperparameter performance
        - prtion models are trained on the current season with optimal hyperparameters for use in bracket creation
    - saves models in `fit_models` dir
11. `model_test.py` - **Not updated**
    - tests models in `fit_models` dir and prints results
12. `winner_predictor.py` - **Not updated**
    - can run to manually predict outcome of a matchup
13. `bracket_generator.py`
    - Manual: archive past year's brackets
    - create new initial bracket for current season's tournament
14. `bracket_scorer.py`
    - Manual: add actual bracket
    - https://www.ncaa.com/brackets/basketball-men/d1/2021
