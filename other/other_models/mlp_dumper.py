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
