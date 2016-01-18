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

