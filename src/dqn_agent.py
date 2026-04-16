"""
DQN Agent for cache replacement policy learning.
Uses deep Q-learning with ε-greedy exploration and experience replay.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


class DenseQNetwork(nn.Module):
    """Dense neural network for Q-value prediction."""
    
    def __init__(self, state_size: int, action_size: int):
        """
        Initialize dense Q-network.
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
        """
        super(DenseQNetwork, self).__init__()
        
        # Dense network architecture with 3-4 hidden layers
        hidden_size = max(64, min(256, state_size))
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for each action
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    """Deep Q-Network agent for cache replacement."""
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 total_steps: int = 50000,
                 device: str = 'cpu'):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions (cache_size)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration probability
            epsilon_end: Minimum exploration probability
            epsilon_decay: Linear decay coefficient (ignored if total_steps provided)
            total_steps: Total training steps (uses linear decay)
            device: 'cpu' or 'cuda'
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.total_steps = total_steps
        self.device = device
        
        # Neural networks (main and target)
        self.q_network = DenseQNetwork(state_size, action_size).to(device)
        self.target_network = DenseQNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Training statistics
        self.training_step = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy strategy.
        
        Args:
            state: Current state
            training: If True, use exploration; if False, use exploitation only
            
        Returns:
            Action index (which item to evict)
        """
        if training and np.random.rand() < self.epsilon:
            # Exploration: random action
            return np.random.randint(0, self.action_size)
        
        # Exploitation: best learned action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()
    
    def train_step(self, states: np.ndarray, actions: np.ndarray,
                   rewards: np.ndarray, next_states: np.ndarray,
                   dones: np.ndarray) -> float:
        """
        Perform one training step on a mini-batch.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            
        Returns:
            Loss value
        """
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states_t).max(dim=1)[0]
            target_q_values = rewards_t + self.gamma * next_q_values * (1 - dones_t)
        
        # Compute loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update training statistics
        self.training_step += 1
        
        # Decay epsilon linearly over total_steps
        progress = self.training_step / self.total_steps
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress
        self.epsilon = max(self.epsilon_end, self.epsilon)  # Ensure minimum
        
        return loss.item()
    
    def update_target_network(self):
        """Sync target network with main network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str):
        """Save network weights."""
        torch.save(self.q_network.state_dict(), filepath)
    
    def load(self, filepath: str):
        """Load network weights."""
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
