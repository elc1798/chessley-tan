import tensorflow as tf
import numpy
import os
import glob
import pickle
import random
import itertools
import scipy.sparse
from sklearn.cross_validation import train_test_split
import h5py
import math
import time

import PROJECT_CONSTANTS as pconst

# God-Tier Mr. Game And Watch RNG: https://youtu.be/wOyKt13HO78?t=39s
RNG = numpy.random

def floatX(x):
    """
    Casts a number to a numpy float32 value as an array

    Params:
        x - floating point number to cast

    Returns:
        x as a NumPy float32 value within a NumPy array
    """
    return numpy.asarray(x, dtype=pconst.FLOAT_TYPE)

def load_data(DIR=pconst.PGN_FILE_DIRECTORY):
    """
    Loads the data from a specified directory containing .hdf5 files obtained
    from .pgn files from game_reader.py

    Params:
        DIR - Directory containing .hdf5 files. PGN_FILE_DIRECTORY (inside
              PROJECT_CONSTANTS) by default

    Yields:
        A generator of read h5py objects
    """
    for fin in glob.glob(DIR + "/*.hdf5"):
        try:
            yield h5py.File(fin, 'r')
        except:
            print("Failed reading file: %s" % (fin))

def get_data(series=["board", "board_rand"]):
    """
    Retrieves data from loaded .hdf5 files from load_data(), and splits them
    into a training set and a testing set. Returns a 2 item array object, with
    index 0 being the training set and index 1 being the testing set

    Params:
        series - Boards to include in sets. Valid board IDs are listed in the
                 store_all_games function in game_reader. They are:
                    board
                    board_rand
                    board_parent
                 By default, the series will contain board and board_rand

    Returns:
        2 item array object in the format [training_set, testing_set]
    """
    data = [[] for i in xrange(len(series))]
    for f in load_data():
        try:
            for index, name in enumerate(series):
                data[index].append(f[name].value)
        except:
            raise
            print("Reading failed on file %s" % (f))

    # Define this function internally
    def stack(vectors):
        """
        Converts a vector value (list) into a NumPy stack

        Params:
            vectors - a list of values

        Returns:
            A vertical or horizontal NumPy stack depending on the shape of the
            vector
        """
        if len(vectors[0].shape) > 1:
            # Vertical stack
            return numpy.vstack(vectors)
        else:
            # Horizontal stack
            return numpy.hstack(vectors)

    # Convert data from list of lists into lists of stacks
    data = [stack(item) for item in data]

    # Split the entries into a training set and a separate test set using scikit
    test_size = pconst.TEST_SIZE_MAX / len(data[0])
    data = train_test_split(*data, test_size=test_size)
    return data

def get_parameters(n_in=None, n_hidden_units=2048, n_hidden_layers=1, weights=None, biases=None):
    """
    Returns a set of TensorFlow Variable weights and a set of TensorFlow
    Variable biases. These are used in our training function

    Notes:
        The Hidden Layer in a neural network is an intermediary layer that
        processes the Input Layer into the Output Layer

    Params:
        n_in - Input Layer
        n_hidden_units - Units for the hidden layer
        n_hidden_layers - Number of hidden layers
        weights - Initial array of weights. If None, get_parameters will
                  generate it
        biases - Initial array of biases. If None, get_parameters will generate
                 it

    Returns:
        A weight set of TensorFlow variables, and a bias set of TensorFlow
        variables
    """
    if weights is None or biases is None:
        if type(n_hidden_units) != list:
            n_hidden_units = [n_hidden_units] * n_hidden_layers
        else:
            n_hidden_layers = len(n_hidden_units)
        weights = []
        biases = []

        # Define internally
        def weight_values(n_in, n_out):
            return numpy.asarray(RNG.uniform(
                low = -numpy.sqrt(6. / (n_in + n_out)),
                high = numpy.sqrt(6. / (n_in + n_out)),
                size = (n_in, n_out)), dtype=pconst.FLOAT_TYPE)

        for i in xrange(n_hidden_layers):
            tmp_in = n_in
            weight = None
            bias = None
            if i != 0:
                tmp_in = n_hidden_units[i - 1]
            if i < n_hidden_layers - 1:
                tmp_out = n_hidden_units[i]
                weight = weight_values(tmp_in, tmp_out)
                bias = numpy.ones(tmp_out, dtype=pconst.FLOAT_TYPE) * pconst.GAMMA
            else:
                weight = numpy.zeros(tmp_in, dtype=pconst.FLOAT_TYPE)
                bias = floatX(0.)
            weights.append(weight)
            biases.append(bias)

    weight_set = [tf.Variable(w) for w in weights]
    bias_set = [tf.Variable(b) for b in biases]

    return weight_set, bias_set

"""
Returns a basic Tensor model for training or testing

Params:
    weight_set - TensorFlow tensor (matrix) of weight values
    bias_set - TensorFlow tensor (matrix) of bias values
    dropout - If True, will remove all neurons that are below threshold

Returns:
    The input layer and output layer (TensorFlow tensor objects)
"""
def get_model(weight_set, bias_set, dropout=False):
    # Create an input layer to process the weights and biases
    input_layer = tf.placeholder(pconst.FLOAT_TYPE, shape=None, name="input_layer")
    # Make a list of dropouts if not already a list
    if type(dropout) != list:
        dropout = [dropout] * len(weight_set)
    # Build a matrix of pieces based on the input layer
    pieces = []
    for piece in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]:
        pieces.append(tf.equal(input_layer, piece))
    # Build the final model (output_layer) from the weights and biases
    binary_layer = tf.concat(axis=1, pieces)
    last_layer = binary_layer
    n = len(weight_set)
    for index in xrange(n - 1):
        intermediary = tf.matmul(last_layer, weight_set[index]) + bias_set[index]
        intermediary = intermediary * (intermediary > 0)
        if dropout[index]:
            mask = numpy.random.binomial(1, 0.5, shape=intermediary.get_shape())
            intermediary = intermediary * tf.cast(mask, pconst.FLOAT_TYPE) * 2

        last_layer = intermediary
    output_layer = tf.matmul(last_layer, weight_set[-1]) + bias_set[-1]
    return input_layer, output_layer

