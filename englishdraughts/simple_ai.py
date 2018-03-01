"""Simple alpha-beta ai to compare against during the training."""
import englishdraughts.core
import time
import hometrainer.agents as agents


class SimpleAI(agents.Agent):
    """Very simple min-max ai to compare against the training progress."""
    def __init__(self):
        pass

    def find_move_with_time_limit(self, game_state, move_time):
        turn_end = time.time() + move_time

        for depth in range(1, 100):
            best_move = self.find_move_with_iteration_limit(game_state, depth)
            if time.time() > turn_end and best_move:
                break

        return best_move

    def find_move_with_iteration_limit(self, game_state, move_iterations):
        child_states = game_state.get_next_game_states()
        our_player = game_state.get_next_player()

        best_move = None
        best_value = -100
        for child_state in child_states:
            child_value = self._state_value(child_state, our_player, move_iterations - 1, -100, 100)

            if child_value > best_value:
                best_move = child_state.get_last_move()
                best_value = child_value

        return best_move

    def _state_value(self, game_state, our_player, depth, alpha, beta):
        child_states = game_state.get_next_game_states()
        if len(child_states) == 0:
            return game_state.calculate_scores()[our_player]

        if depth <= 0:
            total_stones = (game_state.stones_left[englishdraughts.core.PLAYER_ONE] +
                            game_state.stones_left[englishdraughts.core.PLAYER_ONE])
            return (game_state.stones_left[our_player] -
                    game_state.stones_left[englishdraughts.core.opposite_player(our_player)]) / total_stones

        if our_player == game_state.next_player:
            # Maximize
            best_value = -100
            for child_state in child_states:
                child_value = self._state_value(child_state, our_player, depth - 1, alpha, beta)

                if child_value > best_value:
                    best_value = child_value

                if alpha < best_value:
                    alpha = best_value
                if beta <= alpha:
                    break  # Cutoff

            return best_value
        else:
            # Minimize
            best_value = 100
            for child_state in child_states:
                child_value = self._state_value(child_state, our_player, depth - 1, alpha, beta)

                if child_value < best_value:
                    best_value = child_value

                if best_value < beta:
                    beta = best_value
                if beta <= alpha:
                    break  # Cutoff

            return best_value
