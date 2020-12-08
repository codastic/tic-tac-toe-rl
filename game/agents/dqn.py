import random
from copy import deepcopy
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow import keras
from tensorflow.keras import layers

from ..game import Agent
from ..tic_tac_toe import TicTacToeGame, TicTacToeAction, GamePlayer, BOARD_SIZE, BOARD_DIM

def lerp(v, d):
  return v[0] * (1 - d) + v[1] * d

class DQNAgent(Agent):
  def __init__(self, i_agent: int, is_learning: bool = True,
    learning_rate = 1e-3, gamma: float = 0.95,
    epsilon: float = 0.5, epsilon_end: float = 0.001, epsilon_decay_linear: float = 1 / 2000,
    pre_training_games: int = 500, experience_replay_batch_size = 128, memory_size = 10000, 
    reward_draw: float = 5., reward_win: float = 10., reward_loss: float = -10.,
    double_dqn = True, double_dqn_n_games = 1,
    dueling_dqn = True,
    seed = 42):
    super().__init__(i_agent)
    self.gamma = gamma
    self.epsilon = epsilon
    self.epsilon_end = epsilon_end
    self.epsilon_decay_linear = epsilon_decay_linear
    self.learning_rate = learning_rate
    self.experience_replay_batch_size = experience_replay_batch_size
    self.memory_size = memory_size
    self.double_dqn = double_dqn
    self.double_dqn_n_games = double_dqn_n_games
    self.dueling_dqn = dueling_dqn
    self.pre_training_games = pre_training_games
    self.seed = seed
    self.reward_draw = reward_draw
    self.reward_win = reward_win
    self.reward_loss = reward_loss
    self.is_learning = is_learning

    self.num_games = -1
    self.n_inputs = 3 * BOARD_SIZE
    self.n_actions = BOARD_SIZE
    self.memory = deque(maxlen=memory_size)

    random.seed(seed)

    self.model = self.get_model()
    if self.double_dqn:
      self.target_model = self.get_model()
      self.update_target_model() # Sync weights

  def new_game(self, game):
    self.num_games += 1
    self.stage = None
    self.game_log = []

  def end_game(self, game):
    if not self.is_learning:
      return

    self.commit_log(game, True)

    if self.num_games >= self.pre_training_games:
      self.train(self.game_log)

    if (self.experience_replay_batch_size > 0 and
      self.num_games >= self.pre_training_games and 
      len(self.memory) >= self.experience_replay_batch_size):
      samples = random.sample(self.memory, self.experience_replay_batch_size)
      self.train(samples)
    
    self.memory.extend(self.game_log)

    if self.num_games > 0 and self.num_games % self.double_dqn_n_games == 0:
      self.update_target_model()

  def get_reward(self, game: TicTacToeGame) -> int:
    if game.is_game_over():
      winners = game.get_winners()
      if len(winners) > 1:
        return self.reward_draw
      elif winners[0] == self.i_agent:
        return self.reward_win
      else:
        return self.reward_loss
    else:
      return 0

  def get_model(self):
    input_layer = layers.Input((self.n_inputs,))
    layer = input_layer
    layer = layers.Dense(self.n_inputs * self.n_actions, activation='relu', 
      kernel_initializer=keras.initializers.HeUniform(seed=self.seed))(layer)
    
    if self.dueling_dqn: 
      state_value = layers.Dense(1, 
        kernel_initializer=keras.initializers.HeUniform(seed=self.seed))(layer)
      state_value = layers.Lambda(lambda s: keras.backend.expand_dims(s[:, 0], -1), output_shape=(self.n_actions,))(state_value)

      action_advantage = layers.Dense(self.n_actions, 
        kernel_initializer=keras.initializers.HeUniform(seed=self.seed))(layer)
      action_advantage = layers.Lambda(lambda a: a[:, :] - keras.backend.mean(a[:, :], keepdims=True), output_shape=(self.n_actions,))(action_advantage)

      layer = layers.Add()([state_value, action_advantage])
    else:
      layer = layers.Dense(self.n_actions, 
        kernel_initializer=keras.initializers.HeUniform(seed=self.seed))(layer)

    model = keras.Model(inputs=input_layer, outputs=layer)
    model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
      loss='mean_squared_error')

    return model

  def update_target_model(self):
    if self.double_dqn:
      self.target_model.set_weights(self.model.get_weights())

  def get_model_inputs(self, game: TicTacToeGame):
    inputs = keras.utils.to_categorical(game.board, num_classes=3).reshape((self.n_inputs,))
    assert inputs.shape == (self.n_inputs,)
    return inputs

  def get_action_index(self, action: TicTacToeAction):
    return action.position

  def get_action(self, i_action: int):
    return TicTacToeAction(self.i_agent, i_action)

  def get_legal_actions(self, game_state):
    return game_state[0::3]

  def prepare_log(self, game: TicTacToeGame, action: TicTacToeAction):
    if self.is_learning:
      state = self.get_model_inputs(game).copy()
      i_action = self.get_action_index(action)
      self.stage = (state, i_action)

  def commit_log(self, game: TicTacToeGame, done: bool):
    if self.is_learning and self.stage != None:
      state, i_action = self.stage
      next_state = self.get_model_inputs(game).copy()
      reward = self.get_reward(game)
      self.game_log.append((state, i_action, reward, next_state, done))
      self.stage = None

  def train(self, batch):
    """Implements Bellman equation."""
    batch_size = len(batch)
    states = np.zeros((batch_size, self.n_inputs))
    next_states = np.zeros((batch_size, self.n_inputs))
    actions = np.zeros((batch_size), dtype=np.int32)
    rewards = np.zeros(batch_size)
    done = np.zeros(batch_size)

    for i, (state, i_action, reward, next_state, d) in enumerate(batch):
      states[i] = state
      next_states[i] = next_state
      actions[i] = i_action
      rewards[i] = reward
      done[i] = 0. if d else 1.

    assert states.shape == (batch_size, self.n_inputs)
    assert next_states.shape == (batch_size, self.n_inputs)

    q_values = self.model.predict(states)
    q_next = self.model.predict(next_states)

    # Remove illegal next actions
    illegal_value = np.min(q_next) - 1
    legal_actions = np.zeros((batch_size, self.n_actions))
    for i, next_state in enumerate(next_states):
      legal_actions[i] = self.get_legal_actions(next_state)

    q_next = legal_actions * q_next - (legal_actions - 1) * illegal_value

    q_targets = q_values.copy()
    batch_index = np.arange(batch_size, dtype=np.int32)
    if self.double_dqn:
      # Current q network selects the action.
      q_next_actions = np.argmax(q_next, axis=1)
      # Use the target network to evaluate the action.
      q_next_target_model = self.target_model.predict(next_states)
      q_targets[batch_index, actions] = rewards + self.gamma * q_next_target_model[batch_index, q_next_actions] * done
    else:
      q_targets[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * done

    assert q_targets.shape == (batch_size, self.n_actions)

    self.model.fit(states, q_targets, verbose=0)

  def next(self, game: TicTacToeGame) -> bool:
    # Store previous action in action log.
    # We have to wait for the opponent to do its move before storing any rewards
    # that is why we commit here and in end_game().
    self.commit_log(game, False)

    if self.is_learning and (
        self.num_games < self.pre_training_games or 
        random.uniform(0, 1) < lerp([self.epsilon, self.epsilon_end], max(0, self.num_games - self.pre_training_games) * self.epsilon_decay_linear)
      ):
      action = random.choice(game.get_legal_actions(self.i_agent))
    else:
      game_state = self.get_model_inputs(game)
      # Predict action based on current game state.
      q_values = self.model.predict(np.array([game_state]))[0]

      assert q_values.shape == (self.n_actions,)

      # Filter invalid actions
      illegal_value = np.min(q_values) - 1
      legal_actions = self.get_legal_actions(game_state)
      action = self.get_action(np.argmax(legal_actions * q_values  - (legal_actions - 1) * illegal_value))

    self.prepare_log(game, action)
    return game.next(action)