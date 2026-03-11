import pandas as pd
import pickle
import numpy as np

from sklearn import metrics
from sklearn.model_selection import cross_val_score as cvs

from filters import games_up_to_tourney_filter, tourney_filter, games_up_to_tourney_filter, data_for_model, set_up_data
from scraping_utils import check_for_file, read_seasons


if __name__ == "__main__":

    season = read_seasons(seasons_path='seasons_list.txt')[-1]
    source_dir = "3_model_data"
    data = pd.read_pickle(f"{source_dir}/gamelog_exp_clust-{season}.pkl")

    # test models
    Xy_train, Xy_test = data_for_model(data, feature_set='exp_tcf', season=season)
    Xy_train_no_clust, Xy_test_no_clust = data_for_model(data, feature_set='gamelogs', season=season)

    X_train, y_train, X_test, y_test = set_up_data(Xy_train, Xy_test)
    X_train_no_clust, y_train_no_clust, X_test_no_clust, y_test_no_clust = set_up_data(Xy_train_no_clust, Xy_test_no_clust)

    models = {}
    model_paths = [
        f"lr_{season}_fit_model_testing",
        f"rf_{season}_fit_model_testing",
        f"gb_{season}_fit_model_testing",
        f"lr_{season}_fit_model_no_clust_testing",
        f"rf_{season}_fit_model_no_clust_testing",
        f"gb_{season}_fit_model_no_clust_testing"
    ]

    model_dir_path = 'fit_models'

    for model_path in model_paths:
        with open(f"{model_dir_path}/{model_path}.pkl", 'rb') as f:
            pickled_model = pickle.load(f)
            models[model_path] = pickled_model

    for model_name, model in models.items():
        if "no_clust" in model_name:
            Xt, yt, X, y = X_train_no_clust, y_train_no_clust, X_test_no_clust, y_test_no_clust
        else:
            Xt, yt, X, y = X_train, y_train, X_test, y_test

        # cv_score = np.mean(cvs(models, Xt, yt, scoring='accuracy', cv=5))
        y_hat = model.predict(X)
        score = metrics.accuracy_score(y, y_hat)

        print("--------------------------")
        print(f"Model: {model_name}")
        # print(f"CV Score: {cv_score}")
        print(f"Accuracy: {score:2f}")
        print("\n")
