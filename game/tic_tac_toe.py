from enum import IntEnum
from abc import ABC, abstractmethod
import random
import numpy as np
from typing import List
from copy import deepcopy

from .game import Game, Agent, Action

# Tic-Tac-Toe implementation

class GamePlayer(IntEnum):
  EMPTY = 0
  NAUGHT = 1
  CROSS = 2

BOARD_DIM = 3
BOARD_SIZE = BOARD_DIM**2

# Easier to just hardcode the lines we have to check for the winner.
# Board dimensions won't change anyways.
CHECK_LINES = [
  [0, 1, 2], # rows
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6], # cols
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8], # diagonals
  [2, 4, 6]
]

def agent_id_to_char(cell: GamePlayer):
  if (cell == GamePlayer.CROSS):
    return "x"
  if (cell == GamePlayer.NAUGHT):
    return "o"
  return " "

class TicTacToeAction(Action):
  def __init__(self, i_agent: int, position: int):
    super().__init__(i_agent)
    self.position = position

  def is_legal(self, game: 'TicTacToeGame') -> bool:
    return game.board[self.position] == GamePlayer.EMPTY

  def run(self, game: 'TicTacToeGame'):
    assert self.is_legal(game)
    game.board[self.position] = game.players[self.i_agent]

class TicTacToeGame(Game):
  def __init__(self):
    super().__init__(n_agents = 2)
    self.players = [GamePlayer.NAUGHT, GamePlayer.CROSS]
    self.board = np.ndarray(shape=(1, BOARD_SIZE), dtype=int)[0]
    self.board.fill(GamePlayer.EMPTY)

  def is_game_over(self) -> bool:
    for indexes in CHECK_LINES:
      line = [self.board[i] for i in indexes]
      if (line[0] != GamePlayer.EMPTY and line[0] == line[1] == line[2]):
        return True

    for value in self.board:
      if (value == GamePlayer.EMPTY):
        return False

    return True
  
  def get_score(self, i_agent: int) -> int:
    winner = self.get_winner()
    if (winner == self.players[i_agent]):
      return 1
    return 0

  def get_winner(self) -> GamePlayer:
    for indexes in CHECK_LINES:
      line = [self.board[i] for i in indexes]
      if (line[0] != GamePlayer.EMPTY and line[0] == line[1] == line[2]):
        return line[0]
    return None

  def get_legal_actions(self, i_agent: int) -> List[TicTacToeAction]:
    return [TicTacToeAction(i_agent, i) for i in range(len(self.board)) if self.board[i] == GamePlayer.EMPTY]

  def get_hash(self) -> int:
    res = 0
    for i in range(BOARD_SIZE):
        res *= 3
        res += self.board[i]
    return res

  def __str__(self) -> str:
    result = ""
    for y in range(BOARD_DIM):
      if y > 0:
        result += "\n-----\n"
      for x in range(BOARD_DIM):
        if x > 0:
          result += "|"
        result += agent_id_to_char(self.board[x + y * BOARD_DIM])

    return result
