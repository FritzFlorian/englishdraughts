import hometrainer.core
import englishdraughts.core
import time


class SimpleAI:
    """Very simple min-max ai to compare against the training progress."""
    def __init__(self):
        pass

    def find_timed_move(self, game_state, turn_time):
        turn_end = time.time() + turn_time

        for depth in range(1, 100):
            best_move = self.find_move(game_state, depth)
            if time.time() > turn_end and best_move:
                break

        return best_move

    def find_move(self, game_state, depth):
        child_states = game_state.get_next_game_states()
        our_player = game_state.get_next_player()

        best_move = None
        best_value = -100
        for child_state in child_states:
            child_value = self._state_value(child_state, our_player, depth - 1, -100, 100)

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


class SimpleExternalAIEvaluator(hometrainer.core.ExternalEvaluator):
    def __init__(self, nn_client, start_game_state, config):
        super().__init__(nn_client, start_game_state, config)

        self.simple_ai = SimpleAI()

    def external_ai_select_move(self, current_game_state, turn_time):
        return self.simple_ai.find_timed_move(current_game_state, turn_time)


if __name__ == '__main__':
    game_state = englishdraughts.core.DraughtsGameState()

    ai = SimpleAI()
    next_states = game_state.get_next_game_states()
    while len(next_states) > 0:
        if game_state.next_player == englishdraughts.core.PLAYER_ONE:
            move = ai.find_move(game_state, 4)
        else:
            move = ai.find_move(game_state, 1)

        game_state = game_state.execute_move(move)
        next_states = game_state.get_next_game_states()

    print(game_state.stones_left)
    print(game_state.calculate_scores())
