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
Yields a one-time-read generator for the games contained in a PGN file

Params:
    filename - A .pgn file containing valid PGN format chess games

Yields:
    A generator containing the stored games, None otherwise
"""
def read_games(filename):
    f = open(filename)
    while True:
        game = None
        try:
            game = chess.pgn.read_game(f)
        except KeyboardInterrupt:
            raise
        except:
            print("ERROR OCCURRED READING FROM PGN FILE. SKIPPING.")
            continue
        # pgn.read_game returns None if EOF is reached. Break loop on EOF
        if game is None:
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

"""
Returns a known board state from a game as well a different move branch, with
some metadata

Params:
    game - A PyChess format game to be parsed

Returns:
    A tuple with the following indices
    0 : NumPy array of a board state
    1 : NumPy array of parent board state of index 0 state
    2 : NumPy array of state after random move applied onto index 1
    3 : integer storing how many moves from index 1 until checkmate
    4 : Result of game (-1/0/1 for loss/tie/win)
"""
def parse_game_state(game):
    result_id = {
        '1-0': 1,
        '0-1': -1,
        '1/2-1/2': 0
    }
    game_result = game.headers['Result']
    if game_result not in result_id:
        return None
    game_result = result_id[game_result]
    # Build board starting from the end
    board_state = game.end()
    # Exit if the game was incomplete
    if not board_state.board().is_game_over():
        return None
    states = []
    moves_left = 0
    # Line below can be shortened to `while board_state:`
    while board_state is not None: # Redundancy enforced for readability
        # Append a tuple in the format:
        # (num_moves_left, current_board_state, orientation)
        states.append((moves_left, board_state, board_state.board().turn == 0))
        board_state = board_state.parent
        moves_left += 1
    # Remove the initial board state since we know it's a constant and provides
    # no learning value
    states.pop() # or states.pop(len(states) - 1)
    moves_left, board_state, flip = random.choice(states) # Grab a random board
    b = board_state.board()
    array = board2numpyarray(b, flip=flip)
    b_parent = board_state.parent.board()
    array_parent = board2numpyarray(b_parent, flip=(not flip))
    if flip:
        game_result *= -1
    # Generate a random board by performing a random valid move :D
    move = random.choice(list(b_parent.legal_moves))
    b_parent.push(move)
    array_random = board2numpyarray(b_parent, flip=flip)
    return (array, array_parent, array_random, moves_left, game_result)

"""
Stores all games from a PGN file into a HDF5 hierarchical data format file

Params:
    fin - Filename of .pgn file for input
    fout - Filename of output file

Returns:
    None
"""
def store_all_games(fin, fout):
    outfile = h5py.File(fout, 'w')
    # Grab and store values from the return value of parse_game_state
    STATES = [outfile.create_dataset(x, (0, 64), dtype='b', maxshape=(None, 64),
        chunks=True) for x in ['board', 'board_rand', 'board_parent']]
    state_curr, state_rand, state_parent = STATES
    METADATA = [outfile.create_dataset(x, (0,), dtype='b', maxshape=(None,),
        chunks=True) for x in ['result', 'moves_left']]
    res, mvs_left = METADATA
    size = 0
    line = 0
    for game in read_games(fin):
        game = parse_game_state(game)
        if game is None:
            continue
        if line + 1 >= size:
            outfile.flush()
            size = 2 * size + 1
            # Resize H5Py dataset
            [x.resize(size=size, axis=0) for x in (state_curr, state_rand,
                state_parent, res, mvs_left)]
        state_curr[line], state_parent[line], state_rand[line], mvs_left[line], res[line] = game
        line += 1
    [x.resize(size=line, axis=0) for x in (state_curr, state_rand, state_parent,
        res, mvs_left)]
    outfile.close()

