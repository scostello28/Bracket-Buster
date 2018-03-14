# capstone_1

Options:
1. pick a winner

process:
- get team data downloaded and cleaned
  - use Date, Tm, and Opp concatenated as Unique ID to remove duplicates
- make team list
  - put all teams from east in a list then take unique team names and adda ll those to the list... there are 341 teams total...
- set up function to loop through teams
- connect opponents rows for
- test on 2017 season
  - set up test train split on season data
    - using tournament as test set
  - Use KFolds to train model with LogisticRegression
  - How does it do?
- Implement on 2018
  - fill out bracket

- Run on each rounds matchups
- Use to fill out bracket!

Data:
- team gamelog data
- opponents gamelog data
