"""Used by the neural network to convert an evaluation to raw number arrays and vice versa."""
import numpy as np
import englishdraughts.core as core
import hometrainer.util

# Number of values for one hot encoding (field, player_one, player_two)
N_RAW_VALUES = 4 + 4  # Player Stones (x2), Player Queens(x2), Valid Moves in each direction

BOARD_HEIGHT = 8
BOARD_WIDTH = 8


def input(evaluation, calculate_target=False):
    """Converts an evaluation to an number array that is fed to the neural network."""
    normal_evaluation = evaluation.convert_to_normal()
    game_state = normal_evaluation.game_state

    input_array = np.zeros([BOARD_WIDTH, BOARD_HEIGHT, N_RAW_VALUES], dtype=np.int8)
    _add_player_positions(input_array, game_state)
    _add_possible_moves(input_array, game_state)

    if not calculate_target:
        return input_array, None

    value_outputs = np.array([normal_evaluation.get_expected_result()[core.PLAYER_ONE]])
    # A move consists of the piece to move and the direction to move into.
    prob_outputs = np.zeros(BOARD_WIDTH * BOARD_HEIGHT * 4)

    for move, prob in normal_evaluation.get_move_probabilities().items():
        prob_outputs[_move_index(move)] = prob

    target_array = np.concatenate((prob_outputs, value_outputs), axis=0)

    return input_array, target_array


def _add_player_positions(input_array, game_state):
    board = game_state.board

    for x in range(BOARD_WIDTH):
        for y in range(BOARD_HEIGHT):
            field = board[y][x]
            if field == core.PLAYER_ONE:
                input_array[y][x][0] = 1
            elif field == core.PLAYER_ONE_QUEEN:
                input_array[y][x][1] = 1
            elif field == core.PLAYER_TWO:
                input_array[y][x][2] = 1
            elif field == core.PLAYER_TWO_QUEEN:
                input_array[y][x][3] = 1


def _add_possible_moves(input_array, game_state):
    next_game_states = game_state.get_next_game_states()
    for next_game_state in next_game_states:
        move = next_game_state.last_move
        input_array[move.y_old][move.x_old][4 + move.direction] = 1


def output(evaluation, output_array):
    """Adds the results form an output array into an evaluation."""

    # First filter out invalid moves and reshape the result to be an probability distribution.
    output_array_probabilities = output_array[0:-1]
    filter = np.zeros(BOARD_HEIGHT * BOARD_WIDTH * 4)
    for move, prob in evaluation.get_move_probabilities().items():
        # Only let valid moves pass
        filter[_move_index(move)] = 1
    output_array_probabilities = output_array_probabilities * filter
    output_sum = np.sum(output_array_probabilities)
    output_array_probabilities = output_array_probabilities / output_sum

    # Then fill the evaluation with the resulting values.
    for move, prob in hometrainer.util.deepcopy(evaluation.get_move_probabilities()).items():
        evaluation.get_move_probabilities()[move] = output_array_probabilities[_move_index(move)]

    evaluation.get_expected_result()[core.PLAYER_ONE] = output_array[BOARD_WIDTH * BOARD_HEIGHT * 4]
    evaluation.get_expected_result()[core.PLAYER_TWO] = -output_array[BOARD_WIDTH * BOARD_HEIGHT * 4]

    return evaluation


def _move_index(move):
    return (move.x_old + move.y_old * BOARD_HEIGHT) * 4 + move.direction
