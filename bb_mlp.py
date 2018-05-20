import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
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

def define_nn_mlp_model(X_train, y_train, hl1_neurons, hl2_neurons, learning_rate, momentum):
    ''' defines multi-layer-perceptron neural network '''

    '''initialize model'''
    model = Sequential() # sequence of layers

    '''Network Layer Architecture'''
    num_neurons_in_layer_1 = hl1_neurons
    num_neurons_in_layer_2 = hl2_neurons
    num_inputs = X_train.shape[1] # number of features (50)
    num_classes = 1

    # pdb.set_trace()

    '''Layers'''
    model.add(Dense(units=num_neurons_in_layer_1,
                    input_dim=num_inputs,
                    kernel_initializer='normal',
                    activation='relu'))

    model.add(Dense(units=num_neurons_in_layer_2,
                    input_dim=num_neurons_in_layer_1,
                    kernel_initializer='normal',
                    activation='relu'))

    model.add(Dense(units=num_classes,
                    input_dim=num_neurons_in_layer_2,
                    kernel_initializer='normal',
                    activation='sigmoid'))

    '''Set optimizer as stachastic gradient descent'''
    sgd = SGD(lr=learning_rate, decay=1e-7, momentum=momentum)

    '''Set up backprop/train settings'''
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"] )
    return model

def print_output(model, y_train, y_test, rng_seed):
    '''prints model accuracy results'''
    y_train_pred = model.predict_classes(X_train, verbose=0)
    y_test_pred = model.predict_classes(X_test, verbose=0)
    print('Train Accuracy {:.2f}'.format(metrics.accuracy_score(y_train, y_train_pred)))
    print('Test Accuracy {:.2f}'.format(metrics.accuracy_score(y_test, y_test_pred)))



if __name__ == '__main__':
    rng_seed = 2 # set random number generator seed
    data_filepath = 'final_model_data/gamelog_exp_clust.pkl'
    X_train, y_train, X_test, y_test = load_model_data(data_filepath)
    np.random.seed(rng_seed)

    '''Initialize model'''
    model = define_nn_mlp_model(X_train, y_train, hl1_neurons=20, hl2_neurons=20, learning_rate=.001, momentum=.9)

    '''Train model'''

    model.fit(X_train, y_train, epochs=20, batch_size=10, verbose=0,
              validation_split=0.2) # cross val to estimate test error, can monitor overfitting

    '''Figure out Grid Search!!!'''
    # batch_size = [4500, 5000, 5500]
    # epochs = [10, 50, 100]
    # # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # outer_param_grid = dict(batch_size=batch_size, epochs=epochs)
    # grid = GridSearchCV(estimator=model, param_grid=outer_param_grid, scoring='accuracy', n_jobs=-1, cv=5)
    # grid_result = grid.fit(X_train, y_train)
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    '''Predict'''
    print_output(model, y_train, y_test, rng_seed)

    '''TODO: Finish GridSearch!!!!!!'''

    '''Best results before Grid Search'''
    epochs = 100
    batches = 1000
    val_split = .2
    activation = 'relu'
    layer_1_width = 50
    layer_2_width = 60
