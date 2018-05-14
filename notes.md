Questions:
1. How many games could I predict if I just picked favorites?
2. Alter to point spread predictor (Classification to regression)
3. Compare model to 9 celebs (including Obama) and rank

ToDo:
1. MLP Neural Net
2. odds data (impute missing values)
3. if it helps predictions use for 1st round
4. how many would you pick based on pregame odds?
5. is my model better?
6. Fill out new bracket
7. Fill out other brackets
8. save trained models


Data:
1. SOS per year (done)
2. player data (done)
3. Spread/odds data

Refactoring To Do:
1. Update Scraping script to save data to pickle files throughout process (done)
2. Clustering Transformation script
    - streamline transformation to add to gamelog data (done)
3. Create script to transform and combine data for modeling (done)
    - dir for transformed data
      - add team age feature
      - Data with each rolling ave window with clusters and without
4. Test with various rolling ave windows (done)


5. MLP Neural Net
6. Compare model with 9 celebs/analysts
7. README / Presentation

MVP+++
1. Regression (point Spread)
2. Auto bracket filling functionality
3. GMM (Fuzzy Clustering)
