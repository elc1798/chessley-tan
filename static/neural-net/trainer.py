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

