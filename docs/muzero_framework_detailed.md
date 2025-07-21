# MuZero General – Reinforcement Learning Architecture Overview

This document summarizes the reinforcement learning (RL) architecture of MuZero General, focusing on the core classes, their roles, and the training loop. It is intended for educational purposes (e.g. RL course report) and does not cover multi-threading or multi-GPU details.

## Overall Process & Class Call Tree (ASCII)

```
MuZero (muzero.py)
├── SelfPlayWorker (self_play.py)
│   ├── Game (games/*.py) – (e.g. CartPole, Atari, etc.)  
│   │   – (methods: step, reset, legal_actions, etc.)  
│   └── MCTS (self_play.MCTS) – Monte Carlo Tree Search (methods: run, expand, select, backpropagate)  
├── TrainerWorker (trainer.Trainer)  
│   └── MuZeroNetwork (models.py) – (Representation, Dynamics, Prediction heads)  
├── ReplayBufferWorker (replay_buffer.ReplayBuffer) – (methods: save_game, sample_game, sample_batch)  
├── SharedStorageWorker (shared_storage.SharedStorage) – (methods: get_weights, set_weights, set_info)  
└── (Optional) ReanalyseWorker (replay_buffer.Reanalyse) – (refines value targets using latest network)
```

## Key Classes & Responsibilities

- **MuZero** (muzero.py)  
  – Orchestrates workers (SelfPlay, Trainer, ReplayBuffer, SharedStorage) via Ray.  
  – Manages config (e.g. MuZeroConfig) and checkpoints (weights, optimizer state, training stats).  
  – Key methods: train(), test(), load_model(), diagnose_model().

- **SelfPlay** (self_play.py)  
  – Runs Monte Carlo Tree Search (MCTS) (see self_play.MCTS) to sample trajectories.  
  – Interacts with a Game (e.g. CartPole, Atari) (see games/*.py) to step, reset, and obtain legal actions.  
  – Key methods (SelfPlayWorker): continuous_self_play(), (MCTS): run(), expand(), select(), backpropagate().

- **Trainer** (trainer.py)  
  – Computes loss (value, reward, policy) and applies back-propagation (via TrainerWorker).  
  – Calls MuZeroNetwork (models.py) (e.g. initial_inference, recurrent_inference) to compute targets.  
  – Key methods (TrainerWorker): continuous_update_weights(), (Trainer): update_weights().

- **ReplayBuffer** (replay_buffer.py)  
  – Stores game histories (trajectories) and supports priority sampling (e.g. sample_game, sample_batch).  
  – Key methods (ReplayBufferWorker): save_game(), sample_game(), sample_batch().

- **SharedStorage** (shared_storage.py)  
  – Acts as a central parameter server (e.g. get_weights, set_weights, set_info) for broadcasting the latest network weights and training statistics.

- **MuZeroNetwork** (models.py)  
  – Composed of three "heads":  
  • Representation (fθ) – encodes observation (e.g. image, state) into a latent state (e.g. MuZeroFullyConnectedNetwork.representation, MuZeroResidualNetwork.representation).  
  • Dynamics (gθ) – predicts the next latent state and reward given an action (e.g. MuZeroFullyConnectedNetwork.dynamics, MuZeroResidualNetwork.dynamics).  
  • Prediction (hθ) – outputs policy logits (action probabilities) and value (e.g. MuZeroFullyConnectedNetwork.prediction, MuZeroResidualNetwork.prediction).  
  – Two implementations:  
  – MuZeroFullyConnectedNetwork (FC) – (e.g. for CartPole, LunarLander)  
  – MuZeroResidualNetwork (ResNet) – (e.g. for Atari, board games)  
  – Key methods (AbstractNetwork): initial_inference(), recurrent_inference(), get_weights(), set_weights().

## Neural Network Breakdown

- **Representation (fθ)**  
  – Input: observation (e.g. image, state)  
  – Output: latent state (scaled between [0,1] (see appendix in paper))  
  – (Implemented in MuZeroFullyConnectedNetwork.representation or MuZeroResidualNetwork.representation.)

- **Dynamics (gθ)**  
  – Input: latent state + one-hot encoded action (e.g. MuZeroFullyConnectedNetwork.dynamics, MuZeroResidualNetwork.dynamics)  
  – Output: next latent state (scaled) + reward (logits)  
  – (Implemented in MuZeroFullyConnectedNetwork.dynamics or MuZeroResidualNetwork.dynamics.)

- **Prediction (hθ)**  
  – Input: latent state (from fθ or gθ)  
  – Output: policy logits (action probabilities) + value (logits)  
  – (Implemented in MuZeroFullyConnectedNetwork.prediction or MuZeroResidualNetwork.prediction.)

## Training Loop (Reinforcement Learning Flow)

1. **Self-Play (SelfPlayWorker)**  
  – (SelfPlayWorker.continuous_self_play)  
  – (MCTS.run)  
  – (Game.step, Game.reset, Game.legal_actions)  
  – (ReplayBufferWorker.save_game)  
  – (SharedStorageWorker.get_weights)  
  – (MuZeroNetwork.initial_inference, MuZeroNetwork.recurrent_inference)  
  – (MCTS.expand, MCTS.select, MCTS.backpropagate)  
  – (生成 trajectory (observation, action, reward, …) 并存入 replay buffer.)

2. **ReplayBuffer (ReplayBufferWorker)**  
  – (ReplayBufferWorker.save_game)  
  – (ReplayBufferWorker.sample_game, ReplayBufferWorker.sample_batch)  
  – (按优先级采样 mini-batch (e.g. (obs, action, reward, …) 序列) 供 Trainer 更新。)

3. **Trainer (TrainerWorker)**  
  – (TrainerWorker.continuous_update_weights)  
  – (Trainer.update_weights)  
  – (MuZeroNetwork.initial_inference, MuZeroNetwork.recurrent_inference)  
  – (计算 loss (value, reward, policy) (e.g. MSE, cross entropy) 并反向传播 (back-prop) 更新 θ (e.g. Trainer.update_weights).)

4. **SharedStorage (SharedStorageWorker)**  
  – (SharedStorageWorker.set_weights)  
  – (SharedStorageWorker.get_weights)  
  – (将更新后的权重 (e.g. MuZeroNetwork.get_weights) 推送到 SharedStorageWorker, 再由 SelfPlayWorker 拉取 (e.g. SharedStorageWorker.get_weights) 以继续自博弈。)

5. **(Optional) Reanalyse (ReplayBufferWorker.ReanalyseWorker)**  
  – (ReanalyseWorker (replay_buffer.Reanalyse) 利用最新网络 (e.g. MuZeroNetwork.initial_inference, MuZeroNetwork.recurrent_inference) 重新计算 (refine) value targets (e.g. replay_buffer.Reanalyse.reanalyse).)

## Mapping to RL Concepts

- **Policy**  
  – (MCTS (self_play.MCTS) 利用 MuZeroNetwork (models.py) 的 Prediction (hθ) 输出 (policy logits) 作为 prior (e.g. MCTS.expand, MCTS.select).)

- **Value Function**  
  – (MuZeroNetwork (models.py) 的 Prediction (hθ) 输出 (value logits) (e.g. MuZeroFullyConnectedNetwork.prediction, MuZeroResidualNetwork.prediction).)

- **Model-based Dynamics**  
  – (MuZeroNetwork (models.py) 的 Dynamics (gθ) (e.g. MuZeroFullyConnectedNetwork.dynamics, MuZeroResidualNetwork.dynamics) 预测 next latent state 与 reward.)

- **Planning**  
  – (MCTS (self_play.MCTS) (e.g. MCTS.run, MCTS.expand, MCTS.select, MCTS.backpropagate) 在 latent 空间内进行树搜索 (planning).)

- **Off-policy Memory**  
  – (ReplayBuffer (replay_buffer.ReplayBuffer) (e.g. ReplayBufferWorker.save_game, ReplayBufferWorker.sample_game, ReplayBufferWorker.sample_batch) 存储 (off-policy) 历史数据.)

## Reference Files (Key Code Hints)

- **muzero.py** (MuZero, CPUActor, hyperparameter_search, load_model_menu)  
  – (MuZero.train, MuZero.test, MuZero.load_model, MuZero.diagnose_model)  
  – (CPUActor.get_initial_weights)  
  – (hyperparameter_search, load_model_menu (e.g. for model loading & hyper-opt).)

- **models.py** (MuZeroNetwork, MuZeroFullyConnectedNetwork, MuZeroResidualNetwork, AbstractNetwork, RepresentationNetwork, DynamicsNetwork, PredictionNetwork, ResidualBlock, DownSample, DownsampleCNN, mlp, support_to_scalar, scalar_to_support)  
  – (AbstractNetwork.initial_inference, AbstractNetwork.recurrent_inference, AbstractNetwork.get_weights, AbstractNetwork.set_weights)  
  – (MuZeroFullyConnectedNetwork.representation, MuZeroFullyConnectedNetwork.dynamics, MuZeroFullyConnectedNetwork.prediction)  
  – (MuZeroResidualNetwork.representation, MuZeroResidualNetwork.dynamics, MuZeroResidualNetwork.prediction)  
  – (RepresentationNetwork, DynamicsNetwork, PredictionNetwork (e.g. for ResNet (MuZeroResidualNetwork)).)

- **trainer.py** (Trainer, TrainerWorker)  
  – (TrainerWorker.continuous_update_weights, Trainer.update_weights (e.g. loss计算、反向传播、更新 θ).)

- **self_play.py** (SelfPlay, SelfPlayWorker, MCTS)  
  – (SelfPlayWorker.continuous_self_play, SelfPlayWorker.continuous_self_play (e.g. 自博弈 loop).)  
  – (MCTS.run, MCTS.expand, MCTS.select, MCTS.backpropagate (e.g. Monte Carlo Tree Search).)

- **replay_buffer.py** (ReplayBuffer, ReplayBufferWorker, Reanalyse (ReanalyseWorker))  
  – (ReplayBufferWorker.save_game, ReplayBufferWorker.sample_game, ReplayBufferWorker.sample_batch (e.g. 存储、采样 (优先级) 数据).)  
  – (ReanalyseWorker (replay_buffer.Reanalyse) (e.g. 利用最新网络重新计算 (refine) value targets).)

- **shared_storage.py** (SharedStorage, SharedStorageWorker)  
  – (SharedStorageWorker.get_weights, SharedStorageWorker.set_weights, SharedStorageWorker.set_info (e.g. 参数服务器).)

- **games/*.py** (e.g. CartPole, Atari, …)  
  – (Game (e.g. CartPole.Game, Atari.Game) (e.g. step, reset, legal_actions, …).) 