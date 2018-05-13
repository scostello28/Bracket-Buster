import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import theano
import pdb

from classification_model_dataset_selection import games_up_to_2018_season_filter, season2018_filter, set_up_data_for_model

def load_model_data(pickle_filepath):
    '''loads data'''

    data = pd.read_pickle(pickle_filepath)

    X_train, y_train, X_test, y_test = set_up_data_for_model(data, model_data='cluster')

    X_train = X_train.astype(theano.config.floatX)
    X_test = X_test.astype(theano.config.floatX)
    y_train_ohe = np_utils.to_categorical(y_train)
    # print('\nFirst 5 labels of MNIST y_train (one-hot):\n{}'.format(y_train_ohe[:5]))
    # print()
    return X_train, y_train, X_test, y_test, y_train_ohe

def define_nn_mlp_model(X_train, y_train, hl1_neurons, hl2_neurons, learning_rate, momentum):
    ''' defines multi-layer-perceptron neural network '''
    # available activation functions at:
    # https://keras.io/activations/
    # https://en.wikipedia.org/wiki/Activation_function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
    # there are other ways to initialize the weights besides 'uniform', too

    '''initialize model'''
    model = Sequential() # sequence of layers

    '''Network Layer Architecture'''
    num_neurons_in_layer_1 = hl1_neurons # number of neurons in a layer (is it enough?)
    num_neurons_in_layer_2 = hl2_neurons
    num_inputs = X_train.shape[1] # number of features (50) (keep)
    num_classes = 1 #y_train_ohe.shape[1]  # number of classes, 0-1 (keep)

    # pdb.set_trace()

    '''Layers'''
    model.add(Dense(units=num_neurons_in_layer_1,
                    input_dim=num_inputs,
                    kernel_initializer='uniform',
                    activation='relu')) # is tanh the best activation to use here?

    #  maybe add another dense layer here?  How about some deep learning?!?
    model.add(Dense(units=num_neurons_in_layer_2,
                    input_dim=num_neurons_in_layer_1,
                    kernel_initializer='uniform',
                    activation='relu')) # is tanh the best activation to use here?

    model.add(Dense(units=num_classes,
                    input_dim=num_neurons_in_layer_2,
                    kernel_initializer='uniform',
                    activation='softmax')) # keep softmax as last layer

    '''Set optimizer as stachastic gradient descent'''
    sgd = SGD(lr=learning_rate, decay=1e-7, momentum=momentum) # using stochastic gradient descent (keep)

    '''Set up backprop/train settings'''
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"] ) # (keep)
    return model

def print_output(model, y_train, y_test, rng_seed):
    '''prints model accuracy results'''
    y_train_pred = model.predict_classes(X_train, verbose=0)
    y_test_pred = model.predict_classes(X_test, verbose=0)
    # print('\nRandom number generator seed: {}'.format(rng_seed))
    print('\nFirst 30 labels:      {}'.format(y_train[:30]))
    print('First 30 predictions: {}'.format(y_train_pred[:30]))
    train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print('\nTraining accuracy: {:.2f}'.format(train_acc * 100))
    test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print('Test accuracy: %.2f%%' % (test_acc * 100))
    if test_acc < 0.95:
        print("\nMan, that's a poor test accuracy.")
        print("Can't you get it up to 95%?")
    else:
        print("\nYou've made some improvements, I see...")


if __name__ == '__main__':
    rng_seed = 2 # set random number generator seed
    data_filepath = 'model_data/gamelog_5_exp_clust.pkl'
    X_train, y_train, X_test, y_test, y_train_ohe = load_model_data(data_filepath)
    np.random.seed(rng_seed)

    '''Initialize model'''
    model = define_nn_mlp_model(X_train, y_train, hl1_neurons=50, hl2_neurons=60, learning_rate=.001, momentum=.9)

    '''Train model'''
    # Hmm, the fit uses 5 epochs with a batch_size of 5000.  I wonder if that's best?
    model.fit(X_train, y_train, epochs=100, batch_size=1000, verbose=0,
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
