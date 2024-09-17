import numpy as np
import torch
from collections import deque
import random
from project.dqn_model import DQN

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.model = DQN(state_size, action_size)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        if len(minibatch) == 0:
            return

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for experience in minibatch:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        self.model.optimizer.zero_grad()
        loss = self.model.compute_loss(states, actions, rewards, next_states, dones, self.gamma)
        loss.backward()
        self.model.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_action(self, state, epsilon):
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_size, (1,)).item()
        else:
            with torch.no_grad():
                return torch.argmax(self.model.forward(torch.FloatTensor(state).unsqueeze(0))).item()
