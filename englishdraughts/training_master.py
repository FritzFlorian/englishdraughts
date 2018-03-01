"""Helper script to run the training master."""
import hometrainer.distribution as distribution
import englishdraughts.core
import logging
from englishdraughts.config import CustomConfiguration

format = '[%(asctime)-15s] %(levelname)-10s-> %(message)s'
logging.basicConfig(format=format, level=logging.DEBUG)


def main():
    config = CustomConfiguration()
    config._simulations_per_turn = 156
    config._n_self_play = 70
    config._n_external_eval = 14
    config._n_self_eval = 14
    config._external_evaluation_turn_time = 0.75
    # config._training_batch_size = 256
    config._training_batch_size = 1024
    config._c_puct = 2
    training_master = distribution.TrainingMaster('work_dir', 'englishdraughts.neural_network.SimpleNeuralNetwork',
                                                  [englishdraughts.core.DraughtsGameState()], config=config)
    training_master.run()


if __name__ == '__main__':
    main()
