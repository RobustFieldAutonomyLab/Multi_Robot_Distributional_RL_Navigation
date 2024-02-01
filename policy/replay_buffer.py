import random
from collections import deque
import torch
import copy

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
    
    def add(self,item):
        """Add a new experience to memory."""
        self.memory.append(item)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        samples = random.sample(self.memory, k=self.batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for sample in samples:
            state, action, reward, next_state, done = sample 
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return states, actions, rewards, next_states, dones

    def size(self):
        """Return the current size of internal memory."""
        return len(self.memory)


    

