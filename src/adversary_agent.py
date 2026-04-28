"""
Adversary Agent for adversarial cache replacement policy training.
Generates workloads to fool the protagonist cache agent.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class AdversaryNetwork(nn.Module):
    """Dense neural network for workload generation."""
    
    def __init__(self, state_size: int, num_items: int):
        """
        Initialize adversary network.
        
        Args:
            state_size: Size of state vector from cache
            num_items: Number of items in the dataset
        """
        super(AdversaryNetwork, self).__init__()
        
        # Narrower network than protagonist (generates, doesn't evaluate)
        hidden_size = min(128, state_size)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_items)  # Output: which item to request
        
        self.relu = nn.ReLU()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Logits for each item (probability distribution)
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


class AdversaryAgent:
    """Adversary agent that generates hard workloads."""
    
    def __init__(self, state_size: int, num_items: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 0.5,
                 epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.995,
                 total_steps: int = 50000,
                 device: str = 'cpu'):
        """
        Initialize adversary agent.
        
        Args:
            state_size: Size of cache state vector
            num_items: Total number of items in dataset
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration probability
            epsilon_end: Minimum exploration probability
            epsilon_decay: Decay rate for epsilon
            total_steps: Total training steps (for linear decay)
            device: 'cpu' or 'cuda'
        """
        self.state_size = state_size
        self.num_items = num_items
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.total_steps = total_steps
        self.device = device
        
        # Neural networks
        self.q_network = AdversaryNetwork(state_size, num_items).to(device)
        self.target_network = AdversaryNetwork(state_size, num_items).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Training statistics
        self.training_step = 0
    
    def select_request(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select next request using ε-greedy strategy.
        Goal: Generate requests that minimize protagonist's hit rate.
        
        Args:
            state: Current cache state
            training: If True, use exploration; if False, pure exploitation
            
        Returns:
            Item ID to request
        """
        if training and np.random.rand() < self.epsilon:
            # Exploration: random item
            return np.random.randint(0, self.num_items)
        
        # Exploitation: best learned request
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()
    
    def train_step(self, states: np.ndarray, requests: np.ndarray,
                   rewards: np.ndarray, next_states: np.ndarray,
                   dones: np.ndarray) -> float:
        """
        Perform one training step on a mini-batch.
        Reward = negative (want protagonist to miss).
        
        Args:
            states: Batch of cache states
            requests: Batch of item requests (actions)
            rewards: Batch of rewards (negative = good for adversary)
            next_states: Batch of next states
            dones: Batch of done flags
            
        Returns:
            Loss value
        """
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        requests_t = torch.LongTensor(requests).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states_t).gather(1, requests_t.unsqueeze(1))
        
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
        
        # Linear epsilon decay
        progress = self.training_step / self.total_steps
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress
        self.epsilon = max(self.epsilon_end, self.epsilon)
        
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
