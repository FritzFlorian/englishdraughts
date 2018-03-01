"""Helper script to run the playing slave."""
import hometrainer.distribution as distribution
import logging
import hometrainer.config
from englishdraughts.config import CustomConfiguration

format = '[%(asctime)-15s] %(levelname)-10s-> %(message)s'
logging.basicConfig(format=format, level=logging.DEBUG)


def main():
    config = CustomConfiguration()
    playing_slave = distribution.PlayingSlave('tcp://localhost:{}'.format(hometrainer.config.TRAINING_MASTER_PORT),
                                              config=config)
    playing_slave.run()


if __name__ == '__main__':
    main()
