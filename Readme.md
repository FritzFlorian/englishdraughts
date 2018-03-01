# English Draughts

Sample Implementation of https://github.com/FritzFlorian/hometrainer to show how to use it for a specific game.
This is not tweaked, but can be used to try out the AlphaZero algorithm on an simple example.

## Installation

Clone the repo and use it as your current directory.
Install the needed dependencies (tensorflow and hometrainer):
```
virtualenv -p python3 venv
source venv/bin/activate
pip install .
```

The installation might vary on your system.

## Running

The training is separated into the `training_master.py` and the `playing_slave.py`.
The first one executed and coordinates the training, the second executes selfplay games and internal/external
evaluation games. Please only run this in a trusted network as no security is configured.

The two parts of the training are separated to allow distributed execution of selfplay games on spare hardware
that you got at home.

To run both:
```
source venv/bin/activate
python englishdraughts/training_master.py
python englishdraughts/playing_slave.py
```

This will create an folder called `work_dir` containing results and statistics about the run.
