import unittest
import englishdraughts.core as core


class TestMoveExecution(unittest.TestCase):
    def test_normal_moves(self):
        game_state = core.DraughtsGameState()

        next_states = game_state.get_next_game_states()
        self.assertEqual(len(next_states), 7)

        self.assertEqual(next_states[0].board[3][0], core.PLAYER_ONE)
        self.assertEqual(next_states[0].board[2][1], core.EMPTY)
        self.assertEqual(next_states[0].get_next_player(), core.PLAYER_TWO)
        self.assertEqual(next_states[0].moves_without_capture, 1)

        self.assertEqual(len(next_states[0].get_next_game_states()), 7)

    def test_simple_capture(self):
        game_state = core.DraughtsGameState()
        game_state.board = [
            [core.EMPTY] * 8,
            [core.EMPTY] * 8,
            [core.EMPTY, core.EMPTY, core.EMPTY, core.PLAYER_ONE, core.EMPTY, core.EMPTY, core.EMPTY, core.EMPTY],
            [core.EMPTY, core.EMPTY,  core.PLAYER_TWO, core.EMPTY, core.EMPTY, core.EMPTY, core.EMPTY, core.EMPTY],
            [core.EMPTY] * 8,
            [core.EMPTY] * 8,
            [core.EMPTY] * 8,
            [core.EMPTY] * 8,
        ]
        game_state.stones_left = {core.PLAYER_ONE: 1, core.PLAYER_TWO: 1}

        next_states = game_state.get_next_game_states()

        self.assertEqual(len(next_states), 1)
        self.assertEqual(next_states[0].calculate_scores(), {core.PLAYER_ONE: 1, core.PLAYER_TWO: -1})
        self.assertEqual(next_states[0].board[2][3], core.EMPTY)
        self.assertEqual(next_states[0].board[3][2], core.EMPTY)
        self.assertEqual(next_states[0].board[4][1], core.PLAYER_ONE)

    def test_double_capture(self):
        game_state = core.DraughtsGameState()
        game_state.board = [
            [core.EMPTY] * 8,
            [core.EMPTY] * 8,
            [core.EMPTY, core.EMPTY, core.EMPTY, core.PLAYER_ONE, core.EMPTY, core.EMPTY, core.EMPTY, core.EMPTY],
            [core.EMPTY, core.EMPTY, core.PLAYER_TWO, core.EMPTY, core.PLAYER_TWO, core.EMPTY, core.EMPTY, core.EMPTY],
            [core.EMPTY] * 8,
            [core.EMPTY, core.EMPTY, core.EMPTY, core.EMPTY, core.EMPTY, core.EMPTY, core.PLAYER_TWO, core.EMPTY],
            [core.EMPTY] * 8,
            [core.EMPTY] * 8,
        ]
        game_state.stones_left = {core.PLAYER_ONE: 1, core.PLAYER_TWO: 3}

        next_states = game_state.get_next_game_states()

        self.assertEqual(len(next_states), 2)
        self.assertEqual(next_states[0].stones_left, {core.PLAYER_ONE: 1, core.PLAYER_TWO: 2})
        self.assertEqual(next_states[1].stones_left, {core.PLAYER_ONE: 1, core.PLAYER_TWO: 1})


class TestEvaluationConversion(unittest.TestCase):
    def to_and_from_normal(self):
        game_state = core.DraughtsGameState()

        evaluation = game_state.wrap_in_evaluation()
        evaluation.expected_results = {core.PLAYER_ONE: 1, core.PLAYER_TWO: -1}
        to_normal = evaluation.convert_to_normal()
        self.assertEqual(to_normal.next_player, core.PLAYER_ONE)
        self.assertEqual(to_normal.expected_results, {core.PLAYER_ONE: 1, core.PLAYER_TWO: -1})

        game_state = game_state.get_next_game_states()[0]
        evaluation = game_state.wrap_in_evaluation()
        evaluation.expected_results = {core.PLAYER_ONE: 1, core.PLAYER_TWO: -1}
        to_normal = evaluation.convert_to_normal()
        self.assertEqual(to_normal.next_player, core.PLAYER_ONE)
        self.assertEqual(evaluation.game_state.next_player, core.PLAYER_ONE)
        self.assertEqual(evaluation.game_state.board[0][1], core.PLAYER_TWO)
        self.assertEqual(to_normal.expected_results, {core.PLAYER_ONE: -1, core.PLAYER_TWO: 1})

        from_normal = to_normal.convert_from_normal()
        self.assertEqual(evaluation.game_state.next_player, core.PLAYER_TWO)
        self.assertEqual(to_normal.expected_results, {core.PLAYER_ONE: 1, core.PLAYER_TWO: -1})
