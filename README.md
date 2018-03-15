# Bracket Buster

<!-- ![NCAA or Bust](https://media.giphy.com/media/3o84U3i3nkhYoJOm3K/giphy.gif) -->


## Table of Contents
1. [Dataset](#dataset)
  * [Pre-Processing](#pre-processing)
3. [Modeling](#Modeling)
4. [Convolutional Neural Networks](#convolutional-neural-networks)
    * [Model Architecture](#model-architecture)
    * [Training the Model](#training-the-model)  
    * [Patch Selection](#patch-selection)
    * [Results](#results)
5. [Future Directions](#future-directions)




## Hypothesis:
- I can create a model to predict winners that can build a better bracket than Obama

## Dataset
![Team gamelog pic](pictures/gamelog.png)

### Pre-processing

[code_link](game_df_creator.py)

Data Cleaning:
![Cleaned table pic](pictures/cleaneddate.png)

  - Downloaded game data for all teams from 2013 to 2016 (over 4886 games)
  - Feature Engineering:
   - Create win percentage and rolling average Features
   - Points per game, points against per game, field goal percentage,
     free throw percentage, three-point percentage, rebounds per game,
     offensive rebounds per game, assist per game, blocks per game,
     steals per game, turn overs per game and personal fouls per game
  - generate a unique id by mapping names with formatted names
  - combine data to one row for each match!

## Modeling:
**Basic Logistic Regression**

* Trained and tested on data from games form 2013 to 2017
  * using basic train test split on randomized data
```
Accuracy: 0.79 (% predicted correctly)
Precision: 0.78 (predicted positives % correct)
Recall: 0.79 (% of positives predicted correctly)
```

* Trained on data from games form 2013 to 2017
* Tested on 2018 games

```
Accuracy: 0.78 (% predicted correctly)
Precision: 0.78 (predicted positives % correct)
Recall: 0.77 (% of positives predicted correctly)
```


- cross validation and C optimization

C optimization plots

ROC curve and threshold optimizaiton
profit curve optimization

Accuracy: 0.79 (% predicted correctly)
Precision: 0.78 (predicted positives % correct)
Recall: 0.79 (% of positives predicted correctly)


## Pick a winner feature.



notes:
interesting about data
what model did i choose
how did i decide on complexity


Train on multiple years
test against just 1 year

For Tourney:
- get each teams final game in dataframe
- create function to pit two teams against each other and predict
- team 1 wins or loses when against team 2

## Bracket Illustration

Bracket point system:
- Round 1 (64 teams): 1 point per pick
- Round 2 (32 teams): 2
- Round 3 (16 teams): 4
- Round 4 (8 teams): 8
- Round 5 (Final Four): 16
- Round 6 (Championship): 32

### 2016 Bracket
- 2016 with my picks and Obama's picks
- ‎results by cum points
- ‎sean: points
- ‎obama: points

### 2017 bracket




## Learned-
- Pandas, Pandas, Pandas
 - .rolling, .cumsum
 - ‎def function(row): df.apply( thanks michael
 - ‎mapping with dictionaries
 - worked through ‎a lot of problems


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





Jokes:
- any sports fans out there? No... Well I'm gonna do this anyway
- scatter matrix and correlation matrix
  - What am I supposed to make form these?
  - I am supposed to pick my own features?
  - I came here to teach robots to learn.
  - So when the singularity comes they will call me master

<!-- ![NEO](https://media.giphy.com/media/uvoECTG2uCTrG/giphy.gif) -->



Future:
- ‎lag on rolling aves
- ‎sos for each year and possibly rolling
- ‎other features: pace, stats per 100 possessions, team makeup, offensive rating, defensive rating
- ‎map all team names back to common formatting
