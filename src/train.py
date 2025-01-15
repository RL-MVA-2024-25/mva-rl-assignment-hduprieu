import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
import random

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1024
GAMMA = 0.85
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 0.99 
TAU = 0.005
LR = 1e-3
HIDDEN_DIM = 512
PATH = "project_agent_final.pt" #a modifier dans le rendu

class DQN(nn.Module):
    '''
    Dueling DQN architecture
    '''
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.value_branch = self._branch(state_dim, n_actions)
        self.advantage_branch = self._branch(state_dim, n_actions)

    def _branch(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, output_dim)
        )
    
    def forward(self, x):
        value = self.value_branch(x)
        advantage = self.advantage_branch(x)
        return value + advantage - advantage.mean(dim = 1, keepdim = True)


class ProjectAgent:
    def __init__(self, state_dim=6, n_actions=4):
        """
        Implements a double DQN Agent with a Dueling DQN architecture (HIDDEN_STATE parameter).
        Double DQN is implemented using a target network, updated using a soft update rule (TAU parameter).
        Optimisation is done using Adam and a learning rate scheduler (LR parameter).
        Action selection is done using an epsilon-greedy policy, with epsilon geomtrically decaying over time (EPS parameters).
        """
        # Model
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = GAMMA 

        # Computer
        self.path = PATH
        self.device = device

        # DQN parameters
        self.lr = LR
        self.batch_size = BATCH_SIZE
        self.eps = EPS_START
        self.eps_end = EPS_END
        self.eps_decay = EPS_DECAY 

        # DQN definition
        print(f"Using standard DQN architecture on {self.device}.")
        self.policy_net = DQN(self.state_dim, self.n_actions).to(self.device)
        self.target_net = DQN(self.state_dim, self.n_actions).to(self.device)    
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.replay_memory = ReplayMemory_nodeque(capacity=60000)
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=self.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=300, gamma=0.5)

    def act(self, observation, use_random=False):
        # eps-greedy
        if use_random and random.random() < self.eps:
            return np.random.randint(0, self.n_actions)
        # Forward pass
        state_tens = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        Q_values = self.policy_net(state_tens)
        return Q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.replay_memory) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_memory.sample(self.batch_size)

        states_tens = torch.FloatTensor(states).to(self.device)
        actions_tens = torch.LongTensor(actions).to(self.device)
        rewards_tens = torch.FloatTensor(rewards).to(self.device)
        next_states_tens = torch.FloatTensor(next_states).to(self.device)
        dones_tens = torch.FloatTensor(dones).to(self.device)

        # Compute Q-values
        Q_values = self.policy_net(states_tens)
        Q_values = Q_values.gather(1, actions_tens.unsqueeze(1)).squeeze(1)

        # Double DQN: compute max_next_Q and target value
        with torch.no_grad():
            # Select action using the main Q-network
            next_actions = self.policy_net(next_states_tens).argmax(1)
            # Evaluate Q-value of the selected action using the target network
            max_next_Q = self.target_net(next_states_tens).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # Compute target
            target = rewards_tens + self.gamma * max_next_Q * (1 - dones_tens)

        # MSE loss wrt target
        loss = (Q_values - target) ** 2 #Maybe try Hueber loss
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved at {path}")

    def load(self, path = None):
        if path is None:
            path = self.path
        self.policy_net.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Model loaded from {path}")
    
    def train(self, n_episodes):
        for episode in range(n_episodes):
            # Reset environment
            state, _ = env.reset()
            cummulative_reward = 0.0
            for _ in range(200):
                # Take action and store reaction in replay memory
                action = self.act(state, use_random=True)
                next_state, reward, done, truncated, _info = env.step(action)
                self.replay_memory.push(state, action, reward, next_state, done)
                self.train_step()

                # Soft update target network
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                state = next_state
                cummulative_reward += reward
                if done or truncated:
                    break
            
            if self.eps > self.eps_end:
                self.eps *= self.eps_decay
            self.scheduler.step()

            print(f"Episode {episode:4d}, Reward: {int(cummulative_reward):11d}, eps: {self.eps:.2f}, lr: {self.scheduler.get_last_lr()[0]:.5f}")
        self.save(self.path)

class ReplayMemory_nodeque(object):
    # Optimised implementation of a replay memory buffer, without deque
    def __init__(self, capacity = 50000):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
    def push(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states), np.array(dones, dtype=np.float32))
    def __len__(self):
        return len(self.data)

def main():
    agent = ProjectAgent()
    agent.train(400)

if __name__ == "__main__":
    main()
