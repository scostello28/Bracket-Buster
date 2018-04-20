# Bracket Buster

![NCAA or Bust](https://media.giphy.com/media/3o84U3i3nkhYoJOm3K/giphy.gif)


## Table of Contents
1. [Hypotheses](#Hypotheses)
2. [Dataset](#dataset)
3. [Capstone 1](#Capstone 1)
4. [Pre-Processing](#Pre-processing)
5. [Modeling](#Modeling)
6. [Pick a winner](#Pick-a-winner-feature)
7. [Brackets](#Brackets)
  * [2018 Bracket](#2017-Bracket)
8. [Capstone 2](#Capstone 2)
  * [Updates](#Updates)
9. [Clustering](#Clustering)
10. [New Team Composition Features](#New Team Composition Features)
10. [Modeling with Team Composition](#Modeling with Team Composition)
11. [Bracket with Team Composition](#Bracket with Team Composition)
6. [Future Update](#Future-Updates)


## Hypotheses
Capstone 1: I can create a model to predict winners that can build a better bracket than Obama.

Result: Yes I can! Logistic Regression outperformed RandomForests and Gradient Decent Boosting.

Capstone 2: Can I improve the predictive capabilities of my model by adding team composition features--using player archetype clustering.

Result: I don't know yet????

## Dataset
Gamelogs, Rosters and player stats per 100 possessions for each team from the past 5 years. Retrieved from www.sports-reference.com.

![Team gamelog pic](pictures/gamelog.png)


## Capstone 1:

## Pre-processing

[code_link](game_df_creator.py)

Data Cleaning:

![Cleaned table pic](pictures/cleaneddata.png)

  - Pulled gamelog data for all teams from 2014 to 2018
  - The gamelog data was adapted to get a sense for how teams have been playing up to the matchup. The following features were created for predictions:

![Features](pictures/Features.png)

Additional features were utilized to work with the data.
  - game type column to filter by season and tournament games.
  - unique matchup id by mapping names with formatted names to combine data to one row for each match!

![CorrelationMatrix](pictures/corrmatrix.png)

Hard to tell which features are most important based on visual inspection.  So regularization was utilized to determine most useful features for prediction.    

## Modeling

**Logistic Regression**

* LogisticRegression uses Ridge regularization by default and can be switched to Lasso with an argument.  In this case there was not a significant difference between the two.
  * penalty='l2'  -->   Ridge (default)
  * penalty='l1'  -->   Lasso

* Model was trained and tested, using 5-fold cross validation, on data from games from 2014 through 2017 seasons.

```
Accuracy: 0.68 (% predicted correctly)
Precision: 0.67 (predicted positives % correct)
Recall: 0.66 (% of positives predicted correctly)
f1 Score: 0.66 (weighted average of Precision and Recall)
```

* Model was then tested in the games from the 2018 season (hold out set)

```
Accuracy: 0.67 (% predicted correctly)
Precision: 0.66 (predicted positives % correct)
Recall: 0.66 (% of positives predicted correctly)
f1 Score: 0.66 (weighted average of Precision and Recall)
```

**Coefficients**

Ridge and Lasso error rate was identical and looking at the feature coefficients it is not hard to see why.  It is interesting to see that Lasso did not remove any features.  

![Coefficients](pictures/feature_coefficients.png)

**C-optimization**

In logistic regression the regularization parameter is `C` and is the inverse of regularization strength (`alpha = 1 / C`).  Therefore, C must be positive with lower values resulting in stronger regularization.

~~~python
model = LogisticRegression(C=1)
~~~

![Coptimization](pictures/coptimization.png)  ** Update PIC **

Model hyperparameters were optimized using `GridSearchCV` from sklearn's *model selection* library.  This showed an optimal regularization parameter very close to 1--which is the default and results in no regularization.

~~~python
Cs = list(np.linspace(0.1, 3, 100))
grid_search_results = GridSearchCV(model, param_grid={'C':Cs}, scoring='accuracy', cv=5)
grid_search_results.fit(X_train, y_train)
grid_search_results.best_params_
> {'C': 1.17}
~~~

## Pick-a-winner-feature

An interactive function was created to pit two teams against on another to see the modeled outcome.  A threshold of .5 was used to distinguish a winner from a loser.  

- Using final 2017 season stats for each team the model was trained on games from the previous four years to predict the 2017 bracket.

[Code Link](win_or_lose.py)

A clear winner:
~~~
team1: kansas
team2: iona
kansas wins and iona loses!
kansas has 84% chance to win.
iona has 16% chance to win.
~~~


A close match:
~~~
team1: kansas
team2: north-carolina
kansas wins and north-carolina loses!
kansas has 61% chance to win.
north-carolina has 39% chance to win.
~~~


## Brackets

**Bracket point system:**
- Round 1 (64 teams): 1 point per pick
- Round 2 (32 teams): 2
- Round 3 (16 teams): 4
- Round 4 (8 teams): 8
- Round 5 (Final Four): 16
- Round 6 (Championship): 32


### 2018-Bracket

![Obama's 2018 Bracket](pictures/obama2018bracket.png)

![Modeled 2018 Bracket](pictures/model2018bracket.png)

- ‎Model: 81 points
- ‎Obama: 56 points

![Sad Obama](https://media.giphy.com/media/wnDqiePIdJCA8/giphy.gif)


## Capstone 2:

### Additional Data

Rosters and player stats per 100 possessions, in addition to game logs, for each team from the past 5 years. Retrieved from www.sports-reference.com.

![Roster pic](pictures/roster.png)

![stats per 100 possessions pic](pictures/perposs.png)

### Updates

1. SOS per year
2. Team Experience Level (% upper classmen)
3. Team Composition Clusters

## Clustering

Utilized KMeans Clustering to discover player archetypes based on stats. Visualized with tSNE dimensionality reduction.

Center Archetypes:
Cluster 1: Defensive Center - Rebounding and Blocking
Cluster 2: Offensive Center - Strong in the paint
Cluster 3: Shooting Center - Shoots 3's

Cluster 1 rep:
Cluster 2 rep:
Cluster 3 rep:

Forward Archetypes:
Cluster 1: Deep Forwards - Drops 3's and feeds
Cluster 2: Versatile Forwards - Defends and Shoots
Cluster 3: Supporting Forwards - Short range game and passing

Cluster 1 rep:
Cluster 2 rep:
Cluster 3 rep:

Guard Archetypes:
Cluster 1: Downtown Rebounder - Scrappy 3 point shooter
Cluster 2: Supporting Guard - Strong passer and rebounder
Cluster 3: Combo Guard - Shoots and feeds
Cluster 4: Ball Handler - Feeds and breaks knees

Cluster 1 rep:
Cluster 2 rep:
Cluster 3 rep:
Cluster 3 rep:

* Reps are players most mins in cluster

* Team composition by cluster
pivot table by team and year with player label clusters as columns

Show NCAA champion for last 3 years

## Future-Updates
- Additional Features:
  - ‎pace of play, Other stats that my help with clustering.
- test model with various rolling average windows
- test different models after I learn them
  - MLP Neural Net
