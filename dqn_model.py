import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def compute_loss(self, states, actions, rewards, next_states, dones, gamma):
        q_values = self.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.forward(next_states).max(1)[0]
        target_q_values = rewards + (gamma * next_q_values * (1 - dones))
        loss = self.criterion(q_values, target_q_values)
        return loss
