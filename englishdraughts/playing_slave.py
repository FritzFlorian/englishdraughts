import hometrainer.distribution as distribution
import hometrainer.definitions
import logging
from englishdraughts.config import CustomConfiguration

format = '[%(asctime)-15s] %(levelname)-10s-> %(message)s'
logging.basicConfig(format=format, level=logging.DEBUG)


def main():
    config = CustomConfiguration()
    playing_slave = distribution.PlayingSlave('tcp://localhost:{}'.format(hometrainer.definitions.TRAINING_MASTER_PORT), config=config)
    playing_slave.run()


if __name__ == '__main__':
    main()
