# This implementation will be using Python generators and tuples via () and
# yield. This is because machine learning is memory intensive, and generators
# are one-time read objects that do not store memory, hence allowing us to
# devote more memory to the AI when training.

import chess, chess.pgn
import numpy
import sys
import os
import multiprocessing
import itertools
import random
import h5py

"""
Yields a one-time-read generator for the first game contained in a PGN file

Params:
    filename - A .pgn file containing valid PGN format chess games

Yields:
    A generator containing the first game stored, None otherwise
"""
def read_game(filename):
    f = open(filename)
    game = None
    while True:
        try:
            game = chess.pgn.read_game(f)
        except KeyboardInterrupt:
            raise
        except:
            print("ERROR OCCURRED READING FROM PGN FILE. SKIPPING.")
            continue
        # Break out of loop as soon as valid game is read
        if game is not None:
            break
    yield game

"""
Simple conversion (or basic "hash") function to generate unique ID for chess
pieces.

Params:
    piece_type - integer, denoting piece's chess piece type in PGN format: pawn,
                 knight, rook, etc.
    color - int containing piece's color
"""
def gen_piece_id(piece_type, color):
    return piece_type + color * 7

"""
Converts a PGN chess board from PythonChess library to a NumPy array

Params:
    b - board
    flip - True to flip orientation of board (black on bottom, white on top),
           False by default. Board is flipped upon each player's turn

Returns:
    A NumPy array that represents a board state
"""
def board2numpyarray(b, flip=False):
    x = numpy.zeros(64, dtype=numpy.int8)
    for pos, piece in enumerate(b.pieces):
        if piece != 0:
            color = int(bool(b.occupied_co[chess.BLACK] & chess.BB_SQUARES[pos]))
            col = int(pos % 8)
            row = int(pos / 8)
            # Flip the orientation of board around if specified
            if flip:
                row = 7-row
                color = 1 - color
            piece = gen_piece_id(piece, color) # "Hash" Function for identifying pieces
        x[row * 8 + col] = piece
    return x

