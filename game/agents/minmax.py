import random
from typing import List
from copy import deepcopy

from ..game import Agent, Action, Game

class MinMaxAgent(Agent):
  def __init__(self, i_agent: int):
    super().__init__(i_agent)
    self.cache = {}
    self.cache_other = {}

  def get_from_cache(self, game: Game, i_agent: int) -> (Action, int):
    hash = game.get_hash()
    if i_agent == self.i_agent and hash in self.cache:
      actions, score = self.cache[hash]
      return random.choice(actions), score
    elif hash in self.cache_other:
      actions, score = self.cache_other[hash]
      return random.choice(actions), score
    return None

  def store_in_cache(self, game: Game, i_agent: int, actions: List[Action], score: int):
    hash = game.get_hash()
    if i_agent == self.i_agent:
      self.cache[hash] = (actions, score)
    else:
      self.cache_other[hash] = (actions, score)

  def get_max_action(self, game: Game, i_agent: int) -> (Action, int):
    from_cache = self.get_from_cache(game, i_agent)
    if not from_cache is None:
      return from_cache

    actions = game.get_legal_actions(i_agent)
    
    # Randomize order to let agent choose random action if 
    # there are multiple actions with the same best score.
    random.shuffle(actions)

    best_score = None
    best_actions = []
    for action in actions:
      alt_game = deepcopy(game)
      game_over = alt_game.next(action)
      score = 0
      if game_over:
        winners = alt_game.get_winners()
        if len(winners) > 1:
          score = 0
        elif winners[0] == i_agent:
          score = 1
        else:
          score = -1
      else:
        score = self.get_max_action(alt_game, alt_game.get_current_agent())[1] * -1

      if best_score is None or score > best_score:
        best_score = score
        best_actions = [action]
      elif score == best_score:
        best_actions.append(action)
    
    self.store_in_cache(game, i_agent, best_actions, best_score)
    return random.choice(best_actions), best_score

  def next(self, game: Game) -> bool:
    action, _ = self.get_max_action(game, self.i_agent)
    return game.next(action)