# mnk-game-bot
AI bot utilizing Monte Carlo tree search to play [m,n,k-games](https://en.wikipedia.org/wiki/M,n,k-game) (Tic-tac-toe, Gomoku...) with simple Pygame GUI for Player vs. Bot.

# Implemented stuffs
## Bot algorithms
* Monte Carlo tree search (MCTS) with Single-run parallelization:
```
@inproceedings{cazenave2007parallelization,
  title={On the parallelization of UCT},
  author={Cazenave, Tristan and Jouandeau, Nicolas},
  booktitle={Computer games workshop},
  year={2007}
}
```
## Game configs
* Tic-tac-toe
* Gomoku 7x7
* Gomoku 8x9
* Gomoku 9x9
* Gomoku 11x11
* Gomoku 15x15 (MCTS seems not to be effective yet because of speed problem)
## Cython optimization
* MCTS:
```
For end-game board checking
For getting possible moves on board
For MnkBoard and MnkState classes
For calculating UCB + score (not so big improvement)
```

# How to install
First `git clone https://github.com/realphongha/mnk-game-bot.git` to clone this repo.  
Go inside the repo: `cd mnk-game-bot`

## Install requirements
- Python 3.10
- Requirements:
``` bash
uv sync
```

## Compile Cython file
Run:
```
python setup.py build_ext --inplace 
```

# Play vs Bot
```bash
python pvb.py --cfg path/to/{config_file}.yaml
```

# Bot vs Bot
```bash
python bvb.py --cfg path/to/{config_file}.yaml
```
