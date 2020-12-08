from typing import List, Callable
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from collections import Counter
from IPython import display

from .game import Game, Agent

def play_game(game: Game, agents: List[Agent]) -> List[int]:
  for agent in agents:
    agent.new_game(game)

  while not agents[game.get_current_agent()].next(game):
    pass

  for agent in agents:
    agent.end_game(game)
    
  return game.get_winners()

def play_games(create_game: Callable[[], Game], agents: List[Agent], 
                n_games: int = 10000, debug: bool = False, 
                plot: bool = False, plot_window: int = 100, plot_update_n_games: int = 100) -> List[int]:
  results = []
  for i in range(n_games):
    game = create_game()
    winners = play_game(game, agents)
    if len(winners) > 1:
      results.append(-1)
    else:
      results.append(winners[0])

    if plot and ((i + 1) % plot_update_n_games == 0 or i == n_games - 1):
      display.clear_output(wait=True)
      plot_game_results(results, len(agents), plot_window)
      display.display(plt.gcf())
      plt.clf()

  if debug:
    counts = Counter(results)
    print(
      "After {} games we have draws: {} and wins: {}.".format(
        n_games, 
        "{} ({:.2%})".format(counts[-1], counts[-1] / n_games),
        ", ".join(["{} ({:.2%})".format(counts[i], counts[i] / n_games) for i in range(len(agents))])
      )
    )

  return results

def moving_count(items: List[int], value: int, window: int) -> List[int]:
  count = 0
  results = []
  for i in range(len(items)):
    count += -1 if i - window >= 0 and items[i - window] == value else 0
    count += 1 if items[i] == value else 0
    if i >= window - 1:
      results.append(count / window)
  return results

def plot_game_results(results: List[int], num_agents: int, window: int = 100):
  game_number = range(window, len(results) + 1)
  draws = moving_count(results, -1, window)
  winners = [moving_count(results, i, window) for i in range(num_agents)]
  plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))
  plt.plot(game_number, draws, label='Draw')
  for i, winner in enumerate(winners, start = 1):
    plt.plot(game_number, winner, label='Player ' + str(i) + ' wins')
  plt.ylabel(f'Rate over trailing window of {window} games')
  plt.xlabel('Game number')
  plt.xlim([0, len(results)])
  plt.ylim([0, 1])
  ax = plt.gca()
  ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
  ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
  plt.legend(loc='best')