# MuZero General – Texas Hold'em Integration Guide

This document outlines the integration of a Texas Hold'em (gym-holdem) environment into the MuZero General framework. It is intended for educational purposes (e.g. RL course report) and summarizes the key integration points, new classes, and modifications required.

## Integration Overview

The gym-holdem (Texas Hold'em) environment (e.g. gym.make("Texas Hold'em-v1")) provides an interface (reset, step, render, add_player, etc.) that interacts with MuZero's SelfPlayWorker (self_play.py) (e.g. MCTS (Monte Carlo Tree Search) (self_play.MCTS) (methods: run, expand, select, backpropagate) and Game (games/*.py) (methods: step, reset, legal_actions, etc.)). In order to train a MuZero agent on Texas Hold'em, the following integration points must be addressed.

## Key Integration Points

- **New Game Module (games/texas_holdem.py)**  
  – Implement a new Game class (e.g. TexasHoldemGame) that wraps gym-holdem (e.g. gym.make("Texas Hold'em-v1")), and calls (add_player, reset, step, render, etc.) to interact with the environment.  
  – Define a MuZeroConfig subclass (e.g. TexasHoldemMuZeroConfig) (in games/texas_holdem.py) to configure observation_shape (e.g. (n_seats, n_cards, ...)), action_space (e.g. fold, call, raise), stacked_observations, reward_threshold, max_moves, num_players, stack (e.g. 2000 "chips"), and MuZero training hyperparameters (e.g. training steps, learning rate, batch size, MCTS simulations, network (FC or ResNet), etc.).

- **SelfPlay (self_play.py or new self_play_holdem.py)**  
  – Modify (or new) SelfPlayWorker (e.g. HoldemSelfPlayWorker) (in self_play.py (or self_play_holdem.py)) to call TexasHoldemGame's step, reset, legal_actions (e.g. holdem.safe_actions (community_infos, n_seats)) and handle Texas Hold'em–specific actions (e.g. fold, call, raise) and states (e.g. player hands, community cards, stack, betting info).

- **Neural Network (models.py or new models_holdem.py)**  
  – Modify (or new) MuZeroNetwork (e.g. HoldemMuZeroNetwork) (in models.py (or models_holdem.py)) (e.g. MuZeroFullyConnectedNetwork or MuZeroResidualNetwork) to encode Texas Hold'em observations (e.g. player hands, community cards, stack, betting info) (into observation_shape) and output Texas Hold'em action (e.g. fold, call, raise) policy logits and value logits (e.g. MuZeroFullyConnectedNetwork.prediction, MuZeroResidualNetwork.prediction).

- **Trainer (trainer.py or new trainer_holdem.py)**  
  – Modify (or new) TrainerWorker (e.g. HoldemTrainerWorker) (in trainer.py (or trainer_holdem.py)) (e.g. TrainerWorker.continuous_update_weights, Trainer.update_weights) to compute Texas Hold'em–specific loss (e.g. policy loss (cross entropy between Texas Hold'em action logits (e.g. MuZeroFullyConnectedNetwork.prediction, MuZeroResidualNetwork.prediction) and MCTS (e.g. MCTS.run, MCTS.expand, MCTS.select, MCTS.backpropagate) prior), value loss (MSE between Texas Hold'em final reward (e.g. chip change) and network prediction (e.g. MuZeroFullyConnectedNetwork.prediction, MuZeroResidualNetwork.prediction)), reward loss (MSE between Texas Hold'em per–step reward (e.g. bet, call, raise) and network prediction (e.g. MuZeroFullyConnectedNetwork.dynamics, MuZeroResidualNetwork.dynamics))).

- **ReplayBuffer (replay_buffer.py or new replay_buffer_holdem.py)**  
  – Modify (or new) ReplayBufferWorker (e.g. HoldemReplayBufferWorker) (in replay_buffer.py (or replay_buffer_holdem.py)) (e.g. ReplayBufferWorker.save_game, ReplayBufferWorker.sample_game, ReplayBufferWorker.sample_batch) to store Texas Hold'em–specific trajectories (e.g. (observation, action, reward, ...) sequences, where observation (e.g. player hands, community cards, stack, betting info), action (e.g. fold, call, raise), reward (e.g. chip change, bet, call, raise)).

- **MuZero (muzero.py or new muzero_holdem.py)**  
  – Modify (or new) MuZero (e.g. HoldemMuZero) (in muzero.py (or muzero_holdem.py)) (e.g. MuZero.train, MuZero.test, MuZero.load_model, MuZero.diagnose_model) to call Texas Hold'em game (e.g. TexasHoldemGame, TexasHoldemMuZeroConfig) and launch training (e.g. HoldemSelfPlayWorker, HoldemTrainerWorker, HoldemReplayBufferWorker, SharedStorageWorker, ReanalyseWorker (optional)).

- **Diagnose Model (diagnose_model.py or new diagnose_model_holdem.py)**  
  – Modify (or new) diagnostic tools (e.g. HoldemDiagnoseModel) (in diagnose_model.py (or diagnose_model_holdem.py)) (e.g. MuZero.diagnose_model) to visualize Texas Hold'em model (e.g. latent representation (e.g. MuZeroFullyConnectedNetwork.representation, MuZeroResidualNetwork.representation) of Texas Hold'em observation (e.g. player hands, community cards, stack, betting info), Texas Hold'em action (e.g. fold, call, raise) policy logits (e.g. MuZeroFullyConnectedNetwork.prediction, MuZeroResidualNetwork.prediction), Texas Hold'em final reward (e.g. chip change) value logits (e.g. MuZeroFullyConnectedNetwork.prediction, MuZeroResidualNetwork.prediction), Texas Hold'em per–step reward (e.g. bet, call, raise) reward logits (e.g. MuZeroFullyConnectedNetwork.dynamics, MuZeroResidualNetwork.dynamics)).

- **Documentation (README.md or docs)**  
  – Update (or add) documentation (e.g. README.md, docs) (e.g. "How to Run Texas Hold'em MuZero") (e.g. "python muzero_holdem.py" or "python muzero.py --game texas_holdem --config texas_holdem_config.json").

## Reference Files (Key Code Hints)

- **games/texas_holdem.py** (TexasHoldemGame, TexasHoldemMuZeroConfig)  
  – (TexasHoldemGame (e.g. gym.make("Texas Hold'em-v1"), add_player, reset, step, render, etc.).)  
  – (TexasHoldemMuZeroConfig (e.g. observation_shape, action_space, stacked_observations, reward_threshold, max_moves, num_players, stack, training steps, learning rate, batch size, MCTS simulations, network (FC or ResNet), etc.).)

- **self_play_holdem.py** (HoldemSelfPlayWorker, HoldemMCTS (optional))  
  – (HoldemSelfPlayWorker (e.g. continuous_self_play, (MCTS.run, MCTS.expand, MCTS.select, MCTS.backpropagate), (TexasHoldemGame.step, TexasHoldemGame.reset, TexasHoldemGame.legal_actions (e.g. holdem.safe_actions (community_infos, n_seats))), (ReplayBufferWorker.save_game, SharedStorageWorker.get_weights, MuZeroNetwork.initial_inference, MuZeroNetwork.recurrent_inference)).)

- **models_holdem.py** (HoldemMuZeroNetwork (e.g. HoldemMuZeroFullyConnectedNetwork, HoldemMuZeroResidualNetwork), HoldemAbstractNetwork (optional))  
  – (HoldemMuZeroNetwork (e.g. HoldemMuZeroFullyConnectedNetwork.representation, HoldemMuZeroFullyConnectedNetwork.dynamics, HoldemMuZeroFullyConnectedNetwork.prediction, HoldemMuZeroResidualNetwork.representation, HoldemMuZeroResidualNetwork.dynamics, HoldemMuZeroResidualNetwork.prediction).)

- **trainer_holdem.py** (HoldemTrainerWorker, HoldemTrainer (optional))  
  – (HoldemTrainerWorker (e.g. continuous_update_weights, HoldemTrainer.update_weights (e.g. loss (policy (cross entropy), value (MSE), reward (MSE)), back–prop, update θ)).)

- **replay_buffer_holdem.py** (HoldemReplayBufferWorker, HoldemReplayBuffer (optional))  
  – (HoldemReplayBufferWorker (e.g. save_game, sample_game, sample_batch (e.g. (observation (e.g. player hands, community cards, stack, betting info), action (e.g. fold, call, raise), reward (e.g. chip change, bet, call, raise), ...) sequences)).)

- **muzero_holdem.py** (HoldemMuZero, HoldemCPUActor (optional), HoldemHyperparameterSearch (optional), HoldemLoadModelMenu (optional))  
  – (HoldemMuZero (e.g. HoldemMuZero.train, HoldemMuZero.test, HoldemMuZero.load_model, HoldemMuZero.diagnose_model, orchestrating HoldemSelfPlayWorker, HoldemTrainerWorker, HoldemReplayBufferWorker, SharedStorageWorker, ReanalyseWorker (optional)).)

- **diagnose_model_holdem.py** (HoldemDiagnoseModel (optional))  
  – (HoldemDiagnoseModel (e.g. MuZero.diagnose_model (e.g. latent (e.g. HoldemMuZeroFullyConnectedNetwork.representation, HoldemMuZeroResidualNetwork.representation), policy (e.g. HoldemMuZeroFullyConnectedNetwork.prediction, HoldemMuZeroResidualNetwork.prediction), value (e.g. HoldemMuZeroFullyConnectedNetwork.prediction, HoldemMuZeroResidualNetwork.prediction), reward (e.g. HoldemMuZeroFullyConnectedNetwork.dynamics, HoldemMuZeroResidualNetwork.dynamics))).)

- **texas_holdem_config.json** (Optional JSON config (e.g. observation_shape, action_space, stacked_observations, reward_threshold, max_moves, num_players, stack, training steps, learning rate, batch size, MCTS simulations, network (FC or ResNet), etc.).)
