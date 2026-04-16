"""
Experience Replay Buffer for DQN training.
Stores transitions and samples mini-batches for training.
"""

import numpy as np
from collections import deque
from typing import Tuple


class ExperienceReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, max_size: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            max_size: Maximum number of transitions to store
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def store(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample random mini-batch from buffer.
        
        Args:
            batch_size: Size of mini-batch to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) arrays
        """
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size
