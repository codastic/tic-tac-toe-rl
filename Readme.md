# Deep Q-learning the Tic Tac Toe game

This project contains an implementation of the game Tic Tac Toe with several automated agents:

- **RandomAgent** plays fully random.
- **MinMaxAgent** plays perfectly. Best expected outcome against this agent therefore is a draw. If the agent has multiple best next agents it chooses one of them randomly. This is actually important to explore multiple paths of the game state.
- **DQNAgent** is able to learn while playing against any other agent via reinforcment learning. For more information continue reading.

## Jupyter Notebook

The results of training can be found in the Jupyter notebook `training.ipynb`.

## DQN Agent

The DQN agent utilizes the features:

- One-hot encoded input layer (3 * 9 cells)
- One-hot encoded output layer (9 actions)
- Single hidden layer with 243 neurons
- Exploration with decaying epsilon value
- Training with replay memory
- Double DQN to stabilize weights
- Dueling DQN

The trained weights can be found in `dqnagent-first` and `dqnagent-second`. 
The DQN agent has been trained separately as a first and a second player. 
To play with the agent without training again you can just load the provided model weights:

```
agent = DQNAgent(0, is_learning=False)
agent.model = keras.models.load_model('dqnagent-first')
```

## Dependencies

This project in running on Python 3 with the dependencies specified in `requirements.txt`.