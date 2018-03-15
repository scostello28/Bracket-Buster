# Bracket Buster

![NCAA or Bust](https://media.giphy.com/media/3o84U3i3nkhYoJOm3K/giphy.gif)

Goal:
Pick a winner

Data Cleaning:
- Download game data for all teams
- Feature Engineering:
 - Create win percentage and rolling average Features
 - Points per game, points against per game, field goal percentage, free throw percentage, three-point percentage, rebounds per game, offensive rebounds per game, assist per game, blocks per game, steals per game, turn overs per game and personal fouls per game
- generate a unique id with name mapping
- combine data to one row for each match!

For Tourney:
- get each teams final game in dataframe
- create function to pit two teams against each other and predict
- team 1 wins or loses when against team 2



- set up function to loop through teams
- connect opponents rows for
- test on 2017 season
  - set up test train split on season data
    - using tournament as test set
  - Use KFolds to train model with LogisticRegression
  - How does it do?
- Implement on 2018
  - fill out bracket

create dfs with all tourney teams final stats for

- Run on each rounds matchups
- Use to fill out bracket!

Data:
- team gamelog data
- opponents gamelog data


Jokes:
- scatter matrix and correlation matrix
  - What am I supposed to make form these?
  - I am supposed to pick my own features?
  - I came here to teach robots to learn.
  - So when the singularity comes they will call me master

  ![NEO](https://media.giphy.com/media/uvoECTG2uCTrG/giphy.gif)



Future:
Add features:
 - pace
 - offensive rating
 - defensive rating
