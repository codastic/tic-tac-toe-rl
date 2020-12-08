from enum import IntEnum
from typing import List
from abc import ABC, abstractmethod
import random
import numpy as np

class Game(ABC):
  def __init__(self, n_agents: int):
    super().__init__()

    self.n_agents = n_agents
    self.current_agent = 0

  @abstractmethod
  def is_game_over(self) -> bool:
    pass

  @abstractmethod
  def get_score(self, i_agent: int) -> int:
    pass

  def get_current_agent(self) -> int:
    return self.current_agent

  @abstractmethod
  def get_legal_actions(self, i_agent: int) -> List['Action']:
    pass

  def next_agent(self):
    self.current_agent = (self.current_agent + 1) % self.n_agents

  def next(self, action: 'Action') -> bool:
    assert action.is_legal(self)
    action.run(self)
    self.next_agent()
    return self.is_game_over()

  def get_winners(self) -> List[int]:
    best_score = None
    winners = []
    for i in range(self.n_agents):
      score = self.get_score(i)
      if score == best_score:
        winners.append(i)
      elif best_score is None or score > best_score:
        best_score = score
        winners = [i]
    return winners

  @abstractmethod
  def get_hash(self) -> int:
    """Only necessary for MinMaxAgent."""
    pass

  @abstractmethod
  def __str__(self) -> str:
    pass

class Action(ABC):
  def __init__(self, i_agent: int):
    super().__init__()
    self.i_agent = i_agent

  @abstractmethod
  def is_legal(self, game: Game) -> bool:
    pass

  @abstractmethod
  def run(self, game: Game):
    pass

class Agent(ABC):
  def __init__(self, i_agent: int):
    super().__init__()
    self.i_agent = i_agent

  def new_game(self, game: Game):
    pass

  def end_game(self, game: Game):
    pass
    
  @abstractmethod
  def next(self, game: Game) -> bool:
    pass

  def __str__(self) -> str:
    return str(self.i_agent)
