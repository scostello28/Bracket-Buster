import pickle
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
import theano

def games_up_to_2018_tourney_filter(df):
    '''Filter for games up to 2018 season'''
    notourney2018 = (df['GameType'] != 'tourney2018')
    games_up_to_2018_tourney = df[notourney2018]
    return games_up_to_2018_tourney

def tourney2018_filter(df):
    '''Filter for games up to 2018 tourney'''
    tourney2018cond = (df['GameType'] == 'tourney2018')
    tourney2018 = df[tourney2018cond]
    return tourney2018

def data_for_model(df, clusters=True, odds=False):
    '''
    Inputs: Model DataFrame
    Outputs: Vectors for model
    '''

    games_up_to_2018_tourney = games_up_to_2018_tourney_filter(df)

    if clusters and odds:
        Xy = games_up_to_2018_tourney[['W', 'Wp', 'ppg', 'pApg', 'FGp',
            '3Pp', 'FTp', 'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg',
            'PFpg', 'sos', 'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2',
            'G0', 'G1', 'G2', 'G3', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp',
            'OPFTp', 'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
            'OPTOpg', 'OPPFpg', 'OPsos', 'OPexp_factor', 'OPC0', 'OPC1', 'OPC2',
            'OPF0', 'OPF1', 'OPF2', 'OPG0', 'OPG1', 'OPG2', 'OPG3', 'final_p']]

    elif clusters and not odds:
        Xy = games_up_to_2018_tourney[['W', 'Wp', 'ppg', 'pApg', 'FGp',
            '3Pp', 'FTp', 'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg',
            'PFpg', 'sos', 'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2',
            'G0', 'G1', 'G2', 'G3', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp',
            'OPFTp', 'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
            'OPTOpg', 'OPPFpg', 'OPsos', 'OPexp_factor', 'OPC0', 'OPC1', 'OPC2',
            'OPF0', 'OPF1', 'OPF2', 'OPG0', 'OPG1', 'OPG2', 'OPG3']]
    else:
        Xy = games_up_to_2018_tourney[['W', 'Wp', 'ppg', 'pApg', 'FGp',
            '3Pp', 'FTp', 'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg',
            'PFpg', 'sos', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp',
            'OPFTp', 'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
            'OPTOpg', 'OPPFpg', 'OPsos']]

    return Xy

def set_up_data(data_df):
    '''Set up features and targets'''
    X = data_df.iloc[:, 1:].values
    y = data_df.iloc[:, 0].values

    '''Balance classes'''
    X, y = SMOTE().fit_sample(X, y)

    return X, y

def load_model_data_for_mlp(data_df):
    '''Set up features and targets for mlp'''
    X = data_df.iloc[:, 1:].values
    y = data_df.iloc[:, 0].values
    X = X_train.astype(theano.config.floatX)
    return X, y

def lr_model(X, y):
    '''
    Set up logistic regession pipeline.
    Input: data matricies
    Output: fit model
    '''
    lr_pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                ('model', LogisticRegression(C=0.1, penalty='l1'))])

    lr_pipeline.fit(X, y)

    filename = "fit_models/lr_fit_model_no_clust.pkl"
    with open(filename, 'wb') as f:
        # Write the model to a file.
        pickle.dump(lr_pipeline, f)

def rf_model(X, y):
    '''
    Set up Random Forest Classification pipeline.
    Input: data matricies
    Output: fit model
    '''
    rf_pipeline = Pipeline(steps=[('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                    n_estimators=530, min_samples_leaf=4,
                    min_samples_split=3, max_features='sqrt'))])

    rf_pipeline.fit(X, y)

    filename = "fit_models/rf_fit_model_no_clust.pkl"
    with open(filename, 'wb') as f:
        # Write the model to a file.
        pickle.dump(rf_pipeline, f)

def gb_model(X, y):
    '''
    Set up Random Gradient Boosting Classification pipeline.
    Input: data matricies
    Output: fit model
    '''
    gb_pipeline = Pipeline(steps=[('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(
                learning_rate=0.1, loss='exponential', max_depth=2,
                max_features=None, min_samples_leaf=2, min_samples_split=2,
                n_estimators=100, subsample=0.5))])

    gb_pipeline.fit(X, y)

    filename = "fit_models/gb_fit_model_no_clust.pkl"
    with open(filename, 'wb') as f:
        # Write the model to a file.
        pickle.dump(gb_pipeline, f)

# def define_nn_mlp_model(hl1_neurons, hl1_init, hl1_act, hl1_dropout_rate,
#                         hl2_neurons, hl2_init, hl2_act, hl2_dropout_rate,
#                         hl3_neurons, hl3_init, hl3_act, hl3_dropout_rate,
#                         hl4_init, learning_rate, momentum):
#
#     ''' defines multi-layer-perceptron neural network '''
#
#     '''initialize model'''
#     model = Sequential() # sequence of layers
#
#     '''Network Layer Architecture'''
#     num_neurons_in_layer_1 = hl1_neurons
#     num_neurons_in_layer_2 = hl2_neurons
#     num_neurons_in_layer_3 = hl3_neurons
#     num_inputs = 50 # number of features (50)
#     num_classes = 1
#
#     # pdb.set_trace()
#
#     '''Layers'''
#     model.add(Dense(units=num_neurons_in_layer_1,
#                     input_dim=num_inputs,
#                     kernel_initializer=hl1_init,
#                     activation=hl1_act))
#     model.add(Dropout(hl1_dropout_rate))
#     model.add(Dense(units=num_neurons_in_layer_2,
#                     input_dim=num_neurons_in_layer_1,
#                     kernel_initializer=hl2_init,
#                     activation=hl2_act))
#     model.add(Dropout(hl2_dropout_rate))
#     model.add(Dense(units=num_neurons_in_layer_3,
#                     input_dim=num_neurons_in_layer_2,
#                     kernel_initializer=hl3_init,
#                     activation=hl3_act))
#     model.add(Dropout(hl3_dropout_rate))
#     model.add(Dense(units=num_classes,
#                     input_dim=num_neurons_in_layer_3,
#                     kernel_initializer=hl4_init,
#                     activation='sigmoid'))
#
#     '''Set optimizer as stachastic gradient descent'''
#     sgd = SGD(lr=learning_rate, decay=1e-7, momentum=momentum)
#
#     '''Set up backprop/train settings'''
#     model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"] )
#     return model
#
# def mlp_model(X, y):
#     '''
#     Set up Random Gradient Boosting Classification pipeline.
#     Input: data matricies
#     Output: fit model
#     '''
#     mlp_pipeline = Pipeline(steps=[('scaler', StandardScaler()),
#             ('model', define_nn_mlp_model(hl1_neurons=25, hl1_init='normal',
#             hl1_act='relu', hl1_dropout_rate=0, hl2_neurons=20,
#             hl2_init='normal', hl2_act='relu', hl2_dropout_rate=0,
#             hl3_neurons=15, hl3_init='normal', hl3_act='relu',
#             hl3_dropout_rate=0, hl4_init='normal',
#             learning_rate=0.001, momentum=0.09))])
#
#     mlp_pipeline.fit(X, y, batch_size=20, epochs=50, verbose=1, validation_split=0.2)
#
#     filename = "fit_models/mlp_fit_model_no_clust.pkl"
#     with open(filename, 'wb') as f:
#         # Write the model to a file.
#         pickle.dump(mlp_pipeline, f)


if __name__ == '__main__':

    data = pd.read_pickle('final_model_data/gamelog_exp_clust.pkl')
    no_clust_data = pd.read_pickle('final_model_data/gamelog_exp_clust.pkl')
    odds_data = pd.read_pickle('final_model_data/gamelog_exp_clust_odds.pkl')

    Xy_data = data_for_model(data, clusters=True, odds=False)
    Xy_data_no_clust = data_for_model(no_clust_data, clusters=False, odds=False)
    Xy_data_odds = data_for_model(odds_data, clusters=True, odds=True)


    X, y = set_up_data(Xy_data)
    X_no_clust, y_no_clust = set_up_data(Xy_data_no_clust)
    X_odds, y_odds = set_up_data(Xy_data_odds)

    # print('Data with No Odds')
    # lr_model(X, y)
    # rf_model(X, y)
    # gb_model(X, y)
    # mlp_model(X, y)

    # print('No Clusters or odds')
    # lr_model(X_no_clust, y_no_clust)
    # rf_model(X_no_clust, y_no_clust)
    # gb_model(X_no_clust, y_no_clust)
    # mlp_model(X_no_clust, y_no_clust)

    # print('With Odds')
    # lr_model(X_odds, y_odds)
    # rf_model(X_odds, y_odds)
    # gb_model(X_odds, y_odds)
    # mlp_model(X_odds, y_odds)
