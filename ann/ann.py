# ============================================================================
# Neural Network functions using TensorFlow and Keras
# Author : Valérie Bibeau, Polytechnique Montréal, H4ck4th0n 2023
# ============================================================================

# ---------------------------------------------------------------------------
# Imports
# Database
import pandas as pd
# Arrays for data
import numpy as np
# To normalize data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# NN
np.random.seed(1)               # for reproducibility
import tensorflow
tensorflow.random.set_seed(2)   # for reproducibility
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# ----------------------------------------------------------------------------

def read_data(file_path):
    """Read the database

    Args:
        file_path (string): Name of the path where the database is

    Returns:
        X (array): Features tensor
        Y (array): Outputs tensor
    """

    data = pd.read_csv(file_path, index_col=0)
    data = data.to_numpy()
    X = data[:,:-1]
    Y = data[:,-1].reshape(-1,1)

    return X, Y

def initial_setup(X, Y, test_size, random_state):
    """Set up the training and testing set

    Args:
        X (array): Features tensor
        Y (array): Outputs tensor
        random_state (int): Random number to split the training and testing set

    Returns:
        X and Y: Features and target values of the training and testing set
    """
    # Normalizing features
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    scaler_X.fit(X)
    scaler_Y.fit(Y)
    Xscale = scaler_X.transform(X)
    Yscale = scaler_Y.transform(Y)
    # Split the data into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(Xscale, Yscale, 
                                                        test_size=test_size,
                                                        random_state=random_state)
    
    return X_train, X_test, Y_train, Y_test, scaler_X, scaler_Y

def fit_model(X_train, Y_train,
              units, layers, activation, lr,
              epochs, batch_size, val_frac, l2,
              verbose):
    """Neural Network architecture/model

    Args:
        X_train and Y_train: Features and target values of the training set
        units (int): Number of units of the first hidden layer
        layers (int): Number of layers in the NN
        activation (string): Activation function of the hidden layers
        epochs (int): Number of iterations
        batch_size (float): Number of inputs that is being used for training and updating the weights
        val_frac (float): Fraction of the training that will serve the validation of the model
        l2 (float): Regularization constant        
        verbose (int): Boolean to specify the verbosity of the training

    Returns:
        history: History of the training
        model: Model of the ANN
        model.count_params: Number of parameters (weights) of the network
    """
    # Clear backend
    keras.backend.clear_session()
    # Optimizer
    opt = keras.optimizers.Adamax(learning_rate=lr)
    # Initializer
    ini = keras.initializers.GlorotUniform()
    # Regularizer
    reg = keras.regularizers.l2(l2)
    # Architecture of the Neural Network
    model = Sequential()
    model.add(Dense(units, input_dim=X_train.shape[1], kernel_initializer=ini, kernel_regularizer=reg, activation=activation))
    l = 1
    while l < layers:
        model.add(Dense(units, kernel_initializer=ini, kernel_regularizer=reg, activation=activation))
        l = l + 1
    model.add(Dense(1, kernel_initializer=ini, kernel_regularizer=reg, activation='linear'))
    # Compile
    model.compile(loss='mse', optimizer=opt, metrics=['mse','mae','mape'])
    model.summary()
    # Early stop for validation
    #early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    # Fit
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=val_frac, verbose=verbose)

    return history, model, model.count_params()