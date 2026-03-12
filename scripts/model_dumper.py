import joblib
import pickle
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from filters import games_up_to_tourney_filter, tourney_filter, games_up_to_tourney_filter, data_for_model, set_up_data
from scraping_utils import check_for_file, read_seasons


def lr_model(X, y, model_name, output_dir):
    """
    Set up logistic regession pipeline.
    Input: data matricies
    Output: fit model
    """

    filename = f"{model_name}.joblib"

    if check_for_file(directory=output_dir, filename=filename):
        return

    lr_pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                ('model', LogisticRegression(C=0.1, penalty='l1', solver='liblinear'))])

    print(f"Training: {model_name}")
    lr_pipeline.fit(X, y)

    print(f"Save: {output_dir}/{filename}")
    joblib.dump(lr_pipeline, f"{output_dir}/{filename}")

    # with open(filename, 'wb') as f:
    #     # Write the model to a file.
    #     pickle.dump(lr_pipeline, f)

def rf_model(X, y, model_name, output_dir):
    """
    Set up Random Forest Classification pipeline.
    Input: data matricies
    Output: fit model
    """

    filename = f"{model_name}.joblib"

    if check_for_file(directory=output_dir, filename=filename):
        return
    
    rf_pipeline = Pipeline(steps=[('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                    n_estimators=530, min_samples_leaf=4,
                    min_samples_split=3, max_features='sqrt'))])

    print(f"Training: {model_name}")
    rf_pipeline.fit(X, y)

    print(f"Save: {output_dir}/{filename}")
    joblib.dump(rf_pipeline, f"{output_dir}/{filename}")
    # with open(filename, 'wb') as f:
    #     # Write the model to a file.
    #     pickle.dump(rf_pipeline, f)

def gb_model(X, y, model_name, output_dir):
    """
    Set up Random Gradient Boosting Classification pipeline.
    Input: data matricies
    Output: fit model
    """

    filename = f"{model_name}.joblib"

    if check_for_file(directory=output_dir, filename=filename):
        return

    gb_pipeline = Pipeline(steps=[('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(
                learning_rate=0.1, loss='exponential', max_depth=2,
                max_features=None, min_samples_leaf=2, min_samples_split=2,
                n_estimators=100, subsample=0.5))])

    print(f"Training: {model_name}")
    gb_pipeline.fit(X, y)

    print(f"Save: {output_dir}/{filename}")
    joblib.dump(gb_pipeline, f"{output_dir}/{filename}")
    # with open(filename, 'wb') as f:
    #     # Write the model to a file.
    #     pickle.dump(gb_pipeline, f)



if __name__ == "__main__":

    season = read_seasons(seasons_path='seasons_list.txt')[-1]

    source_dir = "3_model_data"
    data_dir = "/Users/sean/Documents/bracket_buster/data"
    data = pd.read_pickle(f"{data_dir}/{source_dir}/{season}/gamelog_exp_clust-{season}.pkl")

    # test models
    Xy_train_t, Xy_test_t = data_for_model(data, feature_set='exp_tcf', season=season)
    Xy_train_no_clust_t, Xy_test_no_clust_t = data_for_model(data, feature_set='gamelogs', season=season)

    X_train_t, y_train_t, X_test_t, y_test_t = set_up_data(Xy_train_t, Xy_test_t)
    X_train_no_clust_t, y_train_no_clust_t, X_test_no_clust_t, y_test_no_clust_t = set_up_data(Xy_train_no_clust_t, Xy_test_no_clust_t)

    # # print('Data with No Odds')
    # lr_model(X_train_t, y_train_t, f"lr_{season}_fit_model_testing")
    # rf_model(X_train_t, y_train_t, f"rf_{season}_fit_model_testing")
    # gb_model(X_train_t, y_train_t, f"gb_{season}_fit_model_testing")

    # # print('No Clusters or odds')
    # lr_model(X_train_no_clust_t, y_train_no_clust_t, f"lr_{season}_fit_model_no_clust_testing")
    # rf_model(X_train_no_clust_t, y_train_no_clust_t, f"rf_{season}_fit_model_no_clust_testing")
    # gb_model(X_train_no_clust_t, y_train_no_clust_t, f"gb_{season}_fit_model_no_clust_testing")

    # bracket models
    Xy_train, Xy_test = data_for_model(
        data, 
        feature_set='exp_tcf', 
        train_filter=games_up_to_tourney_filter, 
        test_filter=tourney_filter, 
        season=season
        )

    Xy_train_no_clust, Xy_test_no_clust = data_for_model(
        data, 
        feature_set='gamelogs', 
        train_filter=games_up_to_tourney_filter, 
        test_filter=tourney_filter, 
        season=season
        )

    X_train, y_train = set_up_data(Xy_train, Xy_test, bracket=True)
    X_train_no_clust, y_train_no_clust = set_up_data(Xy_train_no_clust, Xy_test_no_clust, bracket=True)

    # print('Data with No Odds')
    lr_model(
        X_train, 
        y_train, 
        f"lr_{season}_fit_model", 
        output_dir=f"{data_dir}/{source_dir}/{season}"
        )
    rf_model(
        X_train, 
        y_train, 
        f"rf_{season}_fit_model", 
        output_dir=f"{data_dir}/{source_dir}/{season}"
        )
    gb_model(
        X_train, 
        y_train, 
        f"gb_{season}_fit_model", 
        output_dir=f"{data_dir}/{source_dir}/{season}"
        )

    # print('No Clusters or odds')
    lr_model(
        X_train_no_clust, 
        y_train_no_clust, 
        f"lr_{season}_fit_model_no_clust", 
        output_dir=f"{data_dir}/{source_dir}/{season}"
        )
    rf_model(
        X_train_no_clust, 
        y_train_no_clust, 
        f"rf_{season}_fit_model_no_clust", 
        output_dir=f"{data_dir}/{source_dir}/{season}"
        )
    gb_model(
        X_train_no_clust, 
        y_train_no_clust, 
        f"gb_{season}_fit_model_no_clust", 
        output_dir=f"{data_dir}/{source_dir}/{season}"
        )
 