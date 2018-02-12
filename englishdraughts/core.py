import hometrainer.core as core
import hometrainer.util


# Players
EMPTY = 0
PLAYER_ONE = 1
PLAYER_TWO = 2
PLAYER_ONE_QUEEN = 3
PLAYER_TWO_QUEEN = 4


def opposite_player(player):
    if player == PLAYER_ONE:
        return PLAYER_TWO
    if player == PLAYER_TWO:
        return PLAYER_ONE
    if player == PLAYER_ONE_QUEEN:
        return PLAYER_TWO_QUEEN
    if player == PLAYER_TWO_QUEEN:
        return PLAYER_ONE_QUEEN

    return player


class DraughtsMove(core.Move):
    """ Defines one Move in English Draughts.

    To fully specify a move we need to know all positions that where visited during it.
    """
    def __init__(self, x_old, y_old, x_new, y_new):
        self.x_old = x_old
        self.y_old = y_old

        if self.x_old < x_new:
            if self.y_old < y_new:
                self.direction = 0
            else:
                self.direction = 1
        else:
            if self.y_old < y_new:
                self.direction = 2
            else:
                self.direction = 3

    def __hash__(self):
        return self.x_old + 10 * self.y_old + 100 * self.direction

    def __eq__(self, other):
        return self.x_old == other.x_old and self.y_old == other.y_old and self.direction == other.direction


class DraughtsGameState(core.GameState):
    BOARD_SIZE = 8
    START_BOARD = [
        [EMPTY, PLAYER_ONE] * 4,
        [PLAYER_ONE, EMPTY] * 4,
        [EMPTY, PLAYER_ONE] * 4,
        [EMPTY] * 8,
        [EMPTY] * 8,
        [PLAYER_TWO, EMPTY] * 4,
        [EMPTY, PLAYER_TWO] * 4,
        [PLAYER_TWO, EMPTY] * 4,
    ]
    MAX_MOVES_WITHOUT_CAPTURE = 25

    def __init__(self):
        self.board = hometrainer.util.deepcopy(self.START_BOARD)

        self.last_move = None
        self.next_player = PLAYER_ONE
        self.stones_left = {PLAYER_ONE: 12, PLAYER_TWO: 12}
        self.moves_without_capture = 0

        # 'multiple' capture moves (capturing more than one stone in one turn)
        # is handled as a 'multi turn' move to allow simpler encoding for the neural network.
        self.force_capture_move_at = None

    def wrap_in_evaluation(self):
        return DraughtsGameEvaluation(self)

    def calculate_scores(self):
        if self.moves_without_capture >= self.MAX_MOVES_WITHOUT_CAPTURE:
            return {PLAYER_ONE: 0, PLAYER_TWO: 0}

        # Player who could not make a move looses
        if self.next_player == PLAYER_TWO:
            return {PLAYER_ONE: 1, PLAYER_TWO: -1}
        else:
            return {PLAYER_ONE: -1, PLAYER_TWO: 1}

    def get_last_move(self):
        return self.last_move

    def execute_move(self, move: DraughtsMove):
        next_states = self.get_next_game_states()

        for next_state in next_states:
            if next_state.get_last_move() == move:
                return next_state

        return None

    def player_number_at(self, x, y):
        if self.board[y][x] == PLAYER_ONE or self.board[y][x] == PLAYER_ONE_QUEEN:
            return PLAYER_ONE
        if self.board[y][x] == PLAYER_TWO or self.board[y][x] == PLAYER_TWO_QUEEN:
            return PLAYER_TWO
        return EMPTY

    def _next_player_queen(self):
        if self.next_player == PLAYER_ONE:
            return PLAYER_ONE_QUEEN
        if self.next_player == PLAYER_TWO:
            return PLAYER_TWO_QUEEN

    def get_next_game_states(self):
        next_states = []
        next_states_with_capture = []
        capture_move = False

        # Do not calculate anything if the game ended already
        if self.moves_without_capture >= self.MAX_MOVES_WITHOUT_CAPTURE:
            return next_states
        if self.stones_left[PLAYER_ONE] == 0 or self.stones_left[PLAYER_TWO] == 0:
            return next_states

        # Handle multiple capture moves
        if self.force_capture_move_at:
            x, y = self.force_capture_move_at

            # Move must start at the next players position
            player = self.player_number_at(x, y)

            # Queens can move backward
            queen_move = False
            if player == PLAYER_ONE_QUEEN or player == PLAYER_TWO_QUEEN:
                queen_move = True

            return self._capture_move_at(x, y, queen_move)

        for y in range(self.BOARD_SIZE):
            even_bit = (y + 1) % 2  # We need to add 1 to every second row
            for x in range(even_bit, 8, 2):
                # Move must start at the next players position
                player = self.player_number_at(x, y)
                if player != self.next_player:
                    continue

                # Queens can move backward
                queen_move = False
                if player == PLAYER_ONE_QUEEN or player == PLAYER_TWO_QUEEN:
                    queen_move = True

                # See what capture moves are possible
                captures = self._capture_move_at(x, y, queen_move)
                if len(captures) > 0:
                    capture_move = True
                    next_states_with_capture = captures + next_states_with_capture

                # Do not waste time on normal moves if a capture is possible anyways
                if not capture_move:
                    next_states = next_states + self._normal_move_at(x, y, queen_move)

        if capture_move:
            results = next_states_with_capture
        else:
            results = next_states

        # Change the next player allowed to move and capture move limits...
        for result in results:
            if not result.force_capture_move_at:
                result.next_player = opposite_player(self.next_player)
                result.moves_without_capture = 0
            if not capture_move:
                result.moves_without_capture = self.moves_without_capture + 1

        return results

    def _capture_move_at(self, x, y, queen_move):
        next_states = []

        x_over, y_over, x_new, y_new = x - 1, y + self._y_direction(), x - 2, y + 2 * self._y_direction()
        self._capture_move_from_to_over(x, y, x_new, y_new, x_over, y_over, queen_move, next_states)

        x_over, y_over, x_new, y_new = x + 1, y + self._y_direction(), x + 2, y + 2 * self._y_direction()
        self._capture_move_from_to_over(x, y, x_new, y_new, x_over, y_over, queen_move, next_states)

        # Moving 'backwards' is only ok if we have a queen
        if queen_move:
            x_over, y_over, x_new, y_new = x - 1, y - self._y_direction(), x - 2, y - 2 * self._y_direction()
            self._capture_move_from_to_over(x, y, x_new, y_new, x_over, y_over, queen_move, next_states)

            x_over, y_over, x_new, y_new = x + 1, y - self._y_direction(), x + 2, y - 2 * self._y_direction()
            self._capture_move_from_to_over(x, y, x_new, y_new, x_over, y_over, queen_move, next_states)

        return next_states

    def _capture_move_from_to_over(self, x, y, x_new, y_new, x_over, y_over, queen_move, result_list):
        if not (self.BOARD_SIZE > x_new >= 0 and self.BOARD_SIZE > y_new >= 0):
            return

        if self.player_number_at(x_over, y_over) != opposite_player(self.next_player):
            return

        if self.board[y_new][x_new] != EMPTY:
            return

        # Execute the jump/capture
        new_state = hometrainer.util.deepcopy(self)
        new_state.board[y_over][x_over] = EMPTY
        new_state.board[y][x] = EMPTY
        if queen_move:
            new_state.board[y_new][x_new] = self._next_player_queen()
        else:
            if y_new == self.BOARD_SIZE - 1 or y_new == 0:
                new_state.board[y_new][x_new] = self._next_player_queen()
            else:
                new_state.board[y_new][x_new] = self.next_player
        new_state.stones_left[opposite_player(self.next_player)] = \
            self.stones_left[opposite_player(self.next_player)] - 1

        # Store last move
        new_state.last_move = DraughtsMove(x, y, x_new, y_new)

        # Check if the jumps end's here
        further_jumps = new_state._capture_move_at(x_new, y_new, queen_move)
        if len(further_jumps) > 0:
            new_state.force_capture_move_at = (x_new, y_new)
        else:
            new_state.force_capture_move_at = None

        result_list.append(new_state)

    def _normal_move_at(self, x, y, queen_move):
        next_states = []

        x_new, y_new = x - 1, y + self._y_direction()
        self._normal_move_from_to(x, y, x_new, y_new, next_states, queen_move)

        x_new, y_new = x + 1, y + self._y_direction()
        self._normal_move_from_to(x, y, x_new, y_new, next_states, queen_move)

        # Moving 'backwards' is only ok if we have a queen
        if queen_move:
            x_new, y_new = x - 1, y - self._y_direction()
            self._normal_move_from_to(x, y, x_new, y_new, next_states, queen_move)

            x_new, y_new = x + 1, y - self._y_direction()
            self._normal_move_from_to(x, y, x_new, y_new, next_states, queen_move)

        return next_states

    def _normal_move_from_to(self, x, y, x_new, y_new, result_list, queen_move):
        if self.BOARD_SIZE > x_new >= 0 and self.BOARD_SIZE > y_new >= 0 and self.board[y_new][x_new] == EMPTY:
            new_state = hometrainer.util.deepcopy(self)
            new_state.board[y][x] = EMPTY
            if queen_move:
                new_state.board[y_new][x_new] = self._next_player_queen()
            else:
                if y_new == self.BOARD_SIZE - 1 or y_new == 0:
                    new_state.board[y_new][x_new] = self._next_player_queen()
                else:
                    new_state.board[y_new][x_new] = self.next_player
            new_state.last_move = DraughtsMove(x, y, x_new, y_new)
            new_state.force_capture_move_at = None
            result_list.append(new_state)

    def _y_direction(self):
        if self.next_player == PLAYER_ONE:
            return 1
        return - 1

    def get_player_list(self):
        return [PLAYER_ONE, PLAYER_TWO]

    def get_next_player(self):
        return self.next_player


class DraughtsGameEvaluation(core.Evaluation):
    def __init__(self, game_state):
        super().__init__(game_state)

        next_states = game_state.get_next_game_states()
        self.move_probabilities = {next_state.get_last_move(): 0 for next_state in next_states}
        self.expected_results = {PLAYER_ONE: 0, PLAYER_TWO: 0}

        self.next_player = game_state.get_next_player()

    def get_move_probabilities(self):
        return self.move_probabilities

    def set_move_probabilities(self, move_probabilities):
        self.move_probabilities = move_probabilities

    def get_expected_result(self):
        return self.expected_results

    def set_expected_result(self, expected_results):
        self.expected_results = expected_results

    def convert_to_normal(self):
        if self.game_state.get_next_player() == PLAYER_ONE:
            return self

        return self._swap_players()

    def convert_from_normal(self):
        if self.game_state.get_next_player() == self.next_player:
            return self

        return self._swap_players()

    def _swap_players(self):
        result = hometrainer.util.deepcopy(self)

        # Game State
        for x in range(DraughtsGameState.BOARD_SIZE):
            for y in range(DraughtsGameState.BOARD_SIZE):
                result.game_state.board[y][x] = opposite_player(self.game_state.board[y][x])
        result.game_state.next_player = opposite_player(self.game_state.get_next_player())

        # Expected Results
        result.expected_results[PLAYER_ONE] = self.expected_results[PLAYER_TWO]
        result.expected_results[PLAYER_TWO] = self.expected_results[PLAYER_ONE]

        return result
