from .alphazero_mnkgame import AlphaZeroMnkGame
from .mcts_mnkgame import MonteCarloTreeSearchMnkGame
from .random_agent_mnkgame import RandomAgentMnkGame


def get_agent(name, cfg, net=None):
    if name == "random":
        return RandomAgentMnkGame()
    elif name == "mcts":
        return MonteCarloTreeSearchMnkGame(**cfg["bot"]["mcts"])
    elif name == "alphazero":
        m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
        agent = AlphaZeroMnkGame(m, n, k, **cfg["bot"]["alphazero"])
        agent.init_net(net)
        return agent
    else:
        raise ValueError(f"Unknown agent: {name}")

