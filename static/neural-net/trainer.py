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
import train_function as tfunc

# God-Tier Mr. Game And Watch RNG: https://youtu.be/wOyKt13HO78?t=39s
RNG = numpy.random

# TensorFlow session variable declaration. Initialize afterwards.
sess = None

def floatX(x):
    """
    Casts a number to a numpy float32 value as an array

    Params:
        x - floating point number to cast

    Returns:
        x as a NumPy float32 value within a NumPy array
    """
    return numpy.asarray(x, dtype=pconst.NP_FLOAT)

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
    weight_set - list of TensorFlow tensor (matrix) of weight values
    bias_set - list of TensorFlow tensor (matrix) of bias values
    dropout - If True, will remove all neurons that are below threshold

Returns:
    The input layer and output layer (TensorFlow tensor objects)
"""
def get_model(weight_set, bias_set, dropout=False):
    # Create an input layer to process the weights and biases
    input_layer = tf.placeholder(pconst.FLOAT_TYPE, shape=[None, 12 * 64])
    # Make a list of dropouts if not already a list
    if type(dropout) != list:
        dropout = [dropout] * len(weight_set)
    # Build a matrix of pieces based on the input layer
    pieces = []
    for piece in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]:
        pieces.append(tf.cast(tf.equal(input_layer, piece), pconst.FLOAT_TYPE))
    # Build the final model (output_layer) from the weights and biases
    binary_layer = tf.concat(1, pieces)
    last_layer = binary_layer
    n = len(weight_set)
    for index in xrange(n - 1):
        intermediary = tf.matmul(last_layer, weight_set[index]) + bias_set[index]
        intermediary = tf.mul(intermediary,
                tf.fill(intermediary.get_shape().as_list(),
                    tf.greater(intermediary,
                        tf.zeros(intermediary.get_shape().as_list(),
                            pconst.FLOAT_TYPE))))
        if dropout[index]:
            mask = numpy.random.binomial(1, 0.5, shape=intermediary.get_shape())
            intermediary = tf.mul(intermediary, tf.cast(mask, pconst.FLOAT_TYPE))
            intermediary = tf.mul(intermediary,
                    tf.fill(intermediary.get_shape().as_list(), 2))

        last_layer = intermediary
    output_layer = tf.matmul(last_layer, weight_set[-1]) + bias_set[-1]
    return input_layer, output_layer

"""
Returns the training model for our training function

Params:
    weight_set - list of TensorFlow tensor (matrix) of weight values
    bias_set - list of TensorFlow tensor (matrix) of bias values
    dropout - If True, will remove all neurons that are below threshold
    multiplier - An independent variable for our training function
    kappa - An independent variable for our training function

Returns:
    A tuple with the following indices:
        0 - board input layer
        1 - board rand input layer
        2 - board parent input layer
        3 - statistical net loss of probability of making a move
        4 - Regularization of weights and bias
        5 - statistical net loss of probability between current board state, and
            after a random move
        6 - statistical net loss of probability between current board state and
            parent
        7 - inverse of index 6
"""
def get_training_model(weight_set, bias_set, dropout=False, multiplier=10.0,kappa=1.0):
    # Build a dual network, one for the real move, one for a fake random move
    # Train on a negative log likelihood of classifying the right move
    # il = input layer, ol = output layer
    board_il, board_ol = get_model(weight_set, bias_set, dropout)
    board_rand_il, board_rand_ol = get_model(weight_set, bias_set, dropout)
    board_parent_il, board_parent_ol = get_model(weight_set, bias_set, dropout)

    rand_diff = board_ol - board_rand_ol
    loss_a = -tf.log(tf.nn.sigmoid(rand_diff)).mean()
    parent_diff = kappa * (board_ol + board_parent_ol)
    loss_b = -tf.log(tf.nn.sigmoid(parent_diff)).mean()
    loss_c = -tf.log(tf.nn.sigmoid(-parent_diff)).mean()

    # Regularization
    reg = 0
    for x in tf.add(weight_set, bias_set):
        reg += multiplier * tf.square(x).mean()
    loss_net = loss_a + loss_b + loss_c
    return board_il, board_rand_il, board_parent_il, loss_net, reg, loss_a, loss_b, loss_c

"""
Computes updates using Nesterov's Accelerated Gradient Descent Momentum
Optimization algorithm

Params:
    loss - data loss
    params - parameter to compute gradient using loss
    learning_rate - learning rate of our neural network
    momentum - Tensor of momentums

Returns:
    A list of tuples containing:
        Component tensors of gradient
        Components of 'params'
        Components of momentum param
        Components of velocity
"""
def nesterov_update(loss, params, learning_rate, momentum):
    updates = []
    gradients = tf.gradients(loss, params)
    # Convert momentum into a matrix
    if type(momentum) is not pconst.TFTENSOR_CLASS:
        momentum = tf.fill([len(params.eval(session=sess)[0])], momentum)
    # Convert the learning rate into a matrix
    if type(learning_rate) is not pconst.TFTENSOR_CLASS:
        learning_rate = tf.fill([len(params.eval(session=sess)[0])], learning_rate)
    # Build the momentums from the gradients and params
    for param_i, gradient_i in (params.eval(session=sess), gradients):
        # Note that zip gives a tuple version of an iterable)
        momentum_param = tf.Variable(floatX(param_i.eval(session=sess) * 0.))
        sess.run(momentum_param.initializer)
        velocity = tf.sub(tf.mul(momentum, momentum_param),
                tf.mul(learning_rate, gradient_i))
        weight = tf.sub(tf.add(param_i, tf.mul(momentum, velocity)),
                tf.mul(learning_rate, gradient_i))
        updates.append((param_i, weight))
        updates.append((momentum_param, velocity))
    return updates

"""
Generate a learning function

Params:
    weight_set - list of TensorFlow tensor values for weights
    bias_set - list of TensorFlow bias values for biases
    dropout - If True, will drop dead or insignificant neurons
    update - If True, will use Nesterov's momentum optimization updates

Returns:
    A function that can be called to train a data set
"""
def get_function(weight_set, bias_set, dropout=False, update=False):
    # Set up variable values
    board_il, board_rand_il, board_parent_il, loss_net, reg, loss_a, loss_b, loss_c = get_training_model(weight_set, bias_set, dropout=dropout)
    objective = loss_net + reg
    learning_rate = tf.placeholder(pconst.FLOAT_TYPE, shape=[])
    momentum = floatX(0.9)
    updates = []

    if update:
        assert(len(weight_set) == len(bias_set))
        params = []
        for i in xrange(len(weight_set)):
            params.append(tf.add(weight_set[i], bias_set[i]))
        updates = nesterov_update(objective, params, learning_rate, momentum)
    func = tfunc.function(
            inputs=[board_il, board_rand_il, board_parent_il, learning_rate],
            outputs=[loss_net, reg, loss_a, loss_b, loss_c],
            updates=updates)
    return func

"""
Trains the model

Params:
    print_boards - If True, will print the boards that are read. Not advised for
                   large number of boards

Returns:
    None
"""
def train(print_boards=False):
    curr_train, curr_test, rand_train, rand_test, parent_train, parent_test = get_data(["board", "board_rand", "board_parent"])
    if print_boards:
        for board in [curr_train[0], parent_train[0]]:
            for row in xrange(8):
                print ' '.join("%d" % x for x in board[(row * 8):((row + 1) * 8)])
            print

    size = 12*64
    hidden_units_dim = [2048] * 3

    weight_set, bias_set = get_parameters(n_in=size,
            n_hidden_units=hidden_units_dim)

    BATCH_SIZE = min(pconst.TRAIN_BATCH_SIZE,
            curr_train.get_shape().as_list()[0])

    train_func = get_function(weight_set, bias_set, update=True, dropout=False)
    test_func = get_function(weight_set, bias_set, update=False, dropout=False)

    min_test_loss = float("inf") # Infinity!
    base_learning_rate = 0.03
    t_i = time.time() # Initial time

    for i in xrange(2000):
        learning_rate = base_learning_rate * math.exp(-(time.time() - t0) / 86400) # 8640 = # of seconds in a day

        batch_index = random.randint(0, int(curr_train.get_shape().as_list()[0] / BATCH_SIZE) - 1)
        lo = batch_index * BATCH_SIZE
        hi = (batch_index + 1) * BATCH_SIZE
        loss_net, reg, loss_a, loss_b, loss_c = train_func(
                curr_train[lo:hi],
                rand_train[lo:hi],
                parent_train[lo:hi],
                learning_rate)
        # Print training info
        print "Iteration: %4d\tLearning rate: %.4f\tLoss: %.4f\tReg: %.4f" % (i,
                learning_rate, loss_net, reg)
        if i % 200 == 0:
            test_loss, test_reg, _, _, _ = test_func(curr_test, rand_test, parent_test, learning_rate)
            if test_loss < best_test_loss:
                print "MINIMUM LOSS: %.4" % (test_loss)
                best_test_loss = test_loss

                print "Dumping upgraded model"
                fout = open('model.chessley', 'w')
                def values(set_x):
                    return [x.eval(session=sess) for x in set_x]
                pickle.dump((values(weight_set), values(bias_set)), fout)
                fout.close()

if __name__ == "__main__":
    train(print_boards=False)

