from hometrainer.config import Configuration
import englishdraughts.simple_ai


class CustomConfiguration(Configuration):
    def __init__(self):
        super().__init__()
        self._n_external_eval = 21

    def external_ai_evaluator(self, nn_client, start_game_state):
        return englishdraughts.simple_ai.SimpleExternalAIEvaluator(nn_client, start_game_state, self)
