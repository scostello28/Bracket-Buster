import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD, Adam
import theano
from sklearn import metrics
import pdb

from model import games_up_to_2018_season_filter, season2018_filter, data_for_model, set_up_data

def load_model_data(pickle_filepath):
    '''loads data'''

    data = pd.read_pickle(pickle_filepath)

    train_df, test_df = data_for_model(data, odds=False)
    X_train, y_train, X_test, y_test = set_up_data(train_df, test_df)

    X_train = X_train.astype(theano.config.floatX)
    X_test = X_test.astype(theano.config.floatX)
    y_train_ohe = np_utils.to_categorical(y_train) # all ready OHE
    # pdb.set_trace()
    return X_train, y_train, X_test, y_test #, y_train_ohe

def define_nn_mlp_model(hl1_neurons, hl1_init, hl1_act, hl1_dropout_rate,
                        hl2_neurons, hl2_init, hl2_act, hl2_dropout_rate,
                        hl3_init, learning_rate, momentum):
    ''' defines multi-layer-perceptron neural network '''

    '''initialize model'''
    model = Sequential() # sequence of layers

    '''Network Layer Architecture'''
    num_neurons_in_layer_1 = hl1_neurons
    num_neurons_in_layer_2 = hl2_neurons
    num_inputs = 50 # number of features (50)
    num_classes = 1

    # pdb.set_trace()

    '''Layers'''
    model.add(Dense(units=num_neurons_in_layer_1,
                    input_dim=num_inputs,
                    kernel_initializer=hl1_init,
                    activation=hl1_act))
    model.add(Dropout(hl1_dropout_rate))
    model.add(Dense(units=num_neurons_in_layer_2,
                    input_dim=num_neurons_in_layer_1,
                    kernel_initializer=hl2_init,
                    activation=hl2_act))
    model.add(Dropout(hl2_dropout_rate))
    model.add(Dense(units=num_classes,
                    input_dim=num_neurons_in_layer_2,
                    kernel_initializer=hl3_init,
                    activation='sigmoid'))

    '''Set optimizer as stachastic gradient descent'''
    sgd = SGD(lr=learning_rate, decay=1e-7, momentum=momentum)
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-7, amsgrad=False)

    '''Set up backprop/train settings'''
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"] )
    return model

def print_output(model, X_train, y_train, X_test, y_test):
    '''prints model accuracy results'''
    y_train_pred = model.predict_classes(X_train, verbose=0)
    y_test_pred = model.predict_classes(X_test, verbose=0)
    print('Train Accuracy {:.3f}'.format(metrics.accuracy_score(y_train, y_train_pred)))
    print('Test Accuracy {:.3f}'.format(metrics.accuracy_score(y_test, y_test_pred)))



if __name__ == '__main__':
    # rng_seed = 2 # set random number generator seed

    data = pd.read_pickle('final_model_data/gamelog_exp_clust.pkl')
    # odds_data = pd.read_pickle('final_model_data/gamelog_exp_clust_odds.pkl')

    train_df, test_df = data_for_model(data, odds=False)
    # odds_train_df, odds_test_df = data_for_model(odds_data, odds=True)

    X_train, y_train, X_test, y_test = set_up_data(train_df, test_df)
    # X_train_odds, y_train_odds, X_test_odds, y_test_odds = set_up_data(odds_train_df, odds_test_df)
    # np.random.seed(rng_seed)

    '''Params'''
    batch_size = 15
    epochs = 20
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # optimizer = sgd
    hl1_neurons = 20
    hl1_initializer = 'normal'
    hl1_activation = 'relu'
    hl1_dropout_rate = 0
    hl2_neurons = 30
    hl2_initializer = 'normal'
    hl2_activation = 'relu'
    hl2_dropout_rate = 0
    hl3_initializer = 'normal'
    learning_rate = .001
    momentum = 0.9

    '''Initialize model'''
    model = define_nn_mlp_model(hl1_neurons=hl1_neurons, hl1_init=hl1_initializer,
        hl1_act=hl1_activation, hl1_dropout_rate=hl1_dropout_rate,
        hl2_neurons=hl2_neurons, hl2_init=hl2_initializer,
        hl2_act=hl2_activation, hl2_dropout_rate=hl2_dropout_rate,
        hl3_init=hl3_initializer,
        learning_rate=learning_rate, momentum=momentum)

    '''Train model'''
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1,
              validation_split=0.2) # cross val to estimate test error, can monitor overfitting

    print_output(model, X_train, y_train, X_test, y_test)

    '''GRID SEARCH'''
    # model = KerasClassifier(build_fn=define_nn_mlp_model, verbose=0)
    #
    # batch_size = [5, 10, 20]
    # epochs = [5, 10, 20]
    # # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # hl1_neurons = [10, 20, 30]
    # hl1_initializer = ['normal']
    # hl1_activation = ['relu', 'sigmoid']
    # hl1_dropout_rate = [0, .25, .5]
    # hl2_neurons = [30, 40]
    # hl2_initializer = ['normal']
    # hl2_activation = ['relu', 'sigmoid']
    # hl2_dropout_rate = [0, .25, .5]
    # hl3_initializer = ['normal']
    # learning_rate = [.001]
    # momentum = [.8, .9]
    #
    # outer_param_grid = dict(batch_size=batch_size, epochs=epochs,
    #     hl1_neurons=hl1_neurons, hl1_init=hl1_initializer,
    #     hl1_act=hl1_activation, hl1_dropout_rate=hl1_dropout_rate,
    #     hl2_neurons=hl2_neurons, hl2_init=hl2_initializer,
    #     hl2_act=hl2_activation, hl2_dropout_rate=hl2_dropout_rate,
    #     hl3_init=hl3_initializer, learning_rate=learning_rate, momentum=momentum)
    #
    # grid = GridSearchCV(estimator=model, param_grid=outer_param_grid,
    #                     scoring='accuracy', n_jobs=-1, cv=5, verbose=1)
    # grid_result = grid.fit(X_train, y_train)
    #
    # print("Best CV score: {:.2f}".format(grid_result.best_score_))
    # print("Best params: {}".format(grid_result.best_params_))
    #
    # '''Predict'''
    # y_test_pred = grid_result.predict(X_test, verbose=0)
    # print('Test Accuracy: {:.2f}'.format(metrics.accuracy_score(y_test, y_test_pred)))
