import random
from ..game import Agent, Game

class RandomAgent(Agent):
  def next(self, game: Game) -> bool:
    action = random.choice(game.get_legal_actions(self.i_agent))
    return game.next(action)