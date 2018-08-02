import pickle
import pandas as pd
import numpy as np

from sklearn import metrics

from filters import data_for_model, set_up_data, post_merge_feature_selection, games_up_to_2018_season_filter, season2018_filter

def model_test(model, model_name, X_test, y_test):
    '''
    Set up logistic regession pipeline.
    Input: train and test matricies
    Output: model predictions and accuracy
    '''

    y_hat = model.predict(X_test)
    score = metrics.accuracy_score(y_test, y_hat)

    print('{}:'.format(model_name))
    print('Test Accuracy: {:.2f}'.format(score))
    # return score

if __name__ == '__main__':
    '''Read in data'''
    data = pd.read_pickle('3_final_model_data/gamelog_exp_tcf.pkl')
    # odds_data = pd.read_pickle('final_model_data/gamelog_exp_tcf_odds.pkl')

    '''Set up data for model'''
    train_df, test_df = data_for_model(data, feature_set='gamelogs', train_filter=games_up_to_2018_season_filter, test_filter=season2018_filter)
    train_df_exp_tcf, test_df_exp_tcf = data_for_model(data, feature_set='exp_tcf', train_filter=games_up_to_2018_season_filter, test_filter=season2018_filter)
    # odds_train_df, odds_test_df = data_for_model(odds_data, odds=True)

    X_train, y_train, X_test, y_test = set_up_data(train_df, test_df)
    X_train_exp_tcf, y_train_exp_tcf, X_test_exp_tcf, y_test_exp_tcf = set_up_data(train_df_exp_tcf, test_df_exp_tcf)

    '''Read in fit models'''
    lr_model = 'fit_models/lr_fit_model.pkl'
    rf_model = 'fit_models/rf_fit_model.pkl'
    gb_model = 'fit_models/gb_fit_model.pkl'

    lr_model_exp_tcf = 'fit_models/lr_fit_model_exp_tcf.pkl'
    rf_model_exp_tcf = 'fit_models/rf_fit_model_exp_tcf.pkl'
    gb_model_exp_tcf = 'fit_models/gb_fit_model_exp_tcf.pkl'

    models = [lr_model, rf_model, gb_model, lr_model_exp_tcf, rf_model_exp_tcf, gb_model_exp_tcf]

    for m in range(len(models)):

        with open(models[m], 'rb') as f:
            model = pickle.load(f)

            if m < 3:
                model_test(model, str(models[m]), X_test, y_test)
            else:
                model_test(model, str(models[m]), X_test_exp_tcf, y_test_exp_tcf)
