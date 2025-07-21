"""
This file contains key code changes needed to integrate Texas Hold'em into MuZero.
It provides example implementations that can be used as reference when modifying the actual source files.
"""

import gym
import holdem
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional

# Constants for Texas Hold'em
NUM_CARDS = 52
NUM_HAND_CARDS = 2
NUM_COMMUNITY_CARDS = 5
MAX_PLAYERS = 6
ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_RAISE = 2
MAX_RAISE_TIMES = 4  # Maximum number of raises in one round

class TexasHoldemGame:
    """Game wrapper for gym-holdem environment."""
    
    def __init__(self, num_players: int = 2):
        self.env = gym.make('TexasHoldem-v1')
        self.num_players = num_players
        self.current_player = 0
        
        # Initialize players
        for seat in range(num_players):
            self.env.add_player(seat, stack=2000)
    
    def _encode_cards(self, cards: List[int]) -> np.ndarray:
        """Encode cards as a binary matrix (rank and suit)."""
        if not cards:
            return np.zeros((13, 4))  # 13 ranks × 4 suits
        
        encoded = np.zeros((13, 4))
        for card in cards:
            rank = card // 4
            suit = card % 4
            encoded[rank, suit] = 1
        return encoded
    
    def _encode_observation(self, player_states, community_info) -> np.ndarray:
        """Encode the game state as a neural network input."""
        (player_infos, player_hands) = zip(*player_states)
        (community_infos, community_cards) = community_info
        
        # Encode player's hand
        hand_encoding = self._encode_cards(player_hands[self.current_player])
        
        # Encode community cards
        community_encoding = self._encode_cards(community_cards)
        
        # Encode betting information
        betting_info = np.array([
            community_infos['pot'],
            community_infos['current_bet'],
            *[info['stack'] for info in player_infos],
            *[info['bet'] for info in player_infos]
        ])
        
        # Combine all information
        return np.concatenate([
            hand_encoding.flatten(),
            community_encoding.flatten(),
            betting_info
        ])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the environment."""
        actions = [ACTION_CALL] * self.num_players  # default action for other players
        actions[self.current_player] = action
        
        # Step the environment
        (player_states, community_info), rewards, done, info = self.env.step(actions)
        
        # Update current player
        self.current_player = (self.current_player + 1) % self.num_players
        
        return self._encode_observation(player_states, community_info), rewards[self.current_player], done, info
    
    def reset(self) -> np.ndarray:
        """Reset the environment."""
        player_states, community_info = self.env.reset()
        self.current_player = 0
        return self._encode_observation(player_states, community_info)
    
    def legal_actions(self) -> List[int]:
        """Return list of legal actions for current player."""
        return holdem.safe_actions(self.env.community_infos, self.num_players)

class TexasHoldemMuZeroConfig:
    """MuZero configuration for Texas Hold'em."""
    
    def __init__(self):
        # Environment config
        self.num_players = 2
        self.observation_shape = (13*4*2 + 13*4 + 2 + 2*MAX_PLAYERS,)  # hands + community + pot/bet + player stacks/bets
        self.action_space = [ACTION_FOLD, ACTION_CALL, ACTION_RAISE]
        self.reward_min = -2000  # max loss (initial stack)
        self.reward_max = 2000   # max win (other player's stack)
        
        # Network config
        self.encoding_size = 256
        self.network = "fullyconnected"  # or "resnet"
        
        # Training config
        self.num_actors = 2
        self.num_simulations = 50
        self.discount = 1.0
        self.batch_size = 128
        self.td_steps = 10
        self.num_training_steps = 1000000
        self.weight_decay = 1e-4
        self.learning_rate = 0.02

class HoldemMuZeroNetwork(nn.Module):
    """Neural network for Texas Hold'em MuZero."""
    
    def __init__(self, config: TexasHoldemMuZeroConfig):
        super().__init__()
        
        self.encoding_size = config.encoding_size
        self.action_space_size = len(config.action_space)
        
        # Representation network (observation -> hidden state)
        self.representation_network = nn.Sequential(
            nn.Linear(np.prod(config.observation_shape), 512),
            nn.ReLU(),
            nn.Linear(512, self.encoding_size),
            nn.ReLU()
        )
        
        # Dynamics network (hidden state + action -> next hidden state + reward)
        self.dynamics_state = nn.Sequential(
            nn.Linear(self.encoding_size + self.action_space_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.encoding_size),
            nn.ReLU()
        )
        
        self.dynamics_reward = nn.Sequential(
            nn.Linear(self.encoding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh()
        )
        
        # Prediction networks (hidden state -> policy + value)
        self.policy = nn.Sequential(
            nn.Linear(self.encoding_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_space_size),
            nn.Softmax(dim=1)
        )
        
        self.value = nn.Sequential(
            nn.Linear(self.encoding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh()
        )
    
    def representation(self, observation: torch.Tensor) -> torch.Tensor:
        """Convert observation to hidden state."""
        return self.representation_network(observation)
    
    def dynamics(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state and reward given current state and action."""
        # One-hot encode action
        action_one_hot = torch.zeros(action.shape[0], self.action_space_size, device=action.device)
        action_one_hot.scatter_(1, action.unsqueeze(1).long(), 1.0)
        
        # Concatenate state and action
        x = torch.cat([state, action_one_hot], dim=1)
        
        # Predict next state and reward
        next_state = self.dynamics_state(x)
        reward = self.dynamics_reward(next_state)
        
        return next_state, reward
    
    def prediction(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict policy and value given state."""
        policy = self.policy(state)
        value = self.value(state)
        return policy, value

def example_usage():
    """Example of how to use the above classes."""
    
    # Create game environment
    game = TexasHoldemGame(num_players=2)
    
    # Create MuZero config
    config = TexasHoldemMuZeroConfig()
    
    # Create neural network
    network = HoldemMuZeroNetwork(config)
    
    # Example interaction
    observation = game.reset()
    
    # Convert observation to tensor
    obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
    
    # Get network predictions
    hidden_state = network.representation(obs_tensor)
    policy, value = network.prediction(hidden_state)
    
    # Select action (e.g., using policy)
    action = policy.argmax(dim=1)
    
    # Step environment
    next_observation, reward, done, info = game.step(action.item())
    
    print(f"Policy: {policy}")
    print(f"Value: {value}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")

if __name__ == "__main__":
    example_usage() 