"""Dense neural network model."""

import numpy as np

import tensorflow as tf
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from keras.backend import clear_session
from keras.optimizers import SGD

from AdamW import AdamW


def create_model(init='glorot_normal', activation_1='relu', activation_2='relu', optimizer='SGD',
                 decay=0.1, n_samples=319):
    """ Define neural network architecture

    Parameters
    ----------
    init : str
        initializer to set inital weights
    activation_1 : str
        activation function to be used in first hidden layer
    activation_2 : str
        activation function to be used in second hidden layer
    optimizer : str
        optimization algorithm to find global minimum on loss surface
    decay : float
        rate of weight decay
    n_samples : int
        number of observations

    Returns
    -------
    model : keras.models.Sequential
        compiled keras model
    """
    clear_session()

    # Determine nodes in hidden layers (Huang et al., 2003)
    m = 1 # number of ouput neurons
    hn_1 = int(np.sum(np.sqrt((m+2)*n_samples)+2*np.sqrt(n_samples/(m+2))))
    hn_2 = int(m*np.sqrt(n_samples/(m+2)))

    # Create layers
    model = Sequential()

    if init == 'glorot_normal':
        init_seeded = initializers.glorot_normal(seed=0)
    elif init == 'glorot_uniform':
        init_seeded = initializers.glorot_uniform(seed=0)

    # Two hidden layers
    if optimizer == 'SGD':
        model.add(Dense(hn_1, input_dim=np.shape(X_new)[1], kernel_initializer=init_seeded,
                        kernel_regularizer=regularizers.l2(decay), activation=activation_1))
        model.add(Dense(hn_2, kernel_initializer=init, kernel_regularizer=regularizers.l2(decay),
                        activation=activation_2))
    elif optimizer == 'AdamW':
        model.add(Dense(hn_1, input_dim=np.shape(X_new)[1], kernel_initializer=init_seeded,
                        activation=activation_1))
        model.add(Dense(hn_2, kernel_initializer=init, activation=activation_2))

    # Output neuron
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))

    # Compile
    if optimizer == 'SGD':
        model.compile(loss='binary_crossentropy', optimizer=SGD(nesterov=True),
                      metrics=["accuracy"])
    elif optimizer == 'AdamW':
        model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=decay),
                      metrics=["accuracy"])

    return model
