from hometrainer.config import Configuration
import englishdraughts.simple_ai


class CustomConfiguration(Configuration):
    """We want to use our own external AI Client to evaluate the playing strength.
    To do this we overwrite the Configuration class. This is not the most elegant way to
    handle this kind of config, but it allows freedom to customize it if needed."""
    def __init__(self):
        super().__init__()
        self._n_external_eval = 21

    def external_ai_agent(self, start_game_state):
        return englishdraughts.simple_ai.SimpleAI()
