import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import *
from utils import *
from config import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.load_model = False

        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 1000000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net and the target net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)
        self.target_net = DQN(action_size)
        self.target_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)

        # initialize target net
        self.update_target_net()

        if self.load_model:
            self.policy_net = torch.load('save_model/breakout_dqn')

    # after some time interval update the target net to be same with policy net
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        # print(state.shape)
        if np.random.rand() <= self.epsilon:
            ### CODE ####
            # Choose a random action
            a = torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)
        else:
            ### CODE ####
            #passes state to policy_net and then picks argmax
            with torch.no_grad():
              state = state[None,:,:]
              a = self.policy_net(torch.tensor(state,device=device, dtype=torch.float)).max(1)[1].view(1, 1)
        return a

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()
        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3] # checks if the game is over

        # this next section heavily follows the outline of the official DQN tutorial, except adapted for numpy,
        # i am citing it https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        not_done = np.array([i for i in range(len(dones)) if dones[i] == False])
        not_done_states = next_states[not_done]

        states = torch.tensor(states,device=device,dtype=torch.float)
        actions = torch.cat(actions)#,device=device,dtype=torch.long)
        rewards = torch.tensor(rewards,device=device,dtype=torch.float)

        # Compute Q(s_t, a) - Q of the current state
        ### CODE ####
        # print('step1')
        state_action_values = self.policy_net(states).gather(1, actions)
        
        # # Find maximum Q-value of action at next state from target net
        # ### CODE ####
        # print('step2')
        next_state_values = torch.zeros(32, device=device)
        next_state_values[not_done] = self.target_net(torch.tensor(not_done_states,device=device)).max(1)[0].detach()
        
        # # Compute Q function of next state
        # ### CODE ####
        # print('step3')
        expected_state_action_values = (next_state_values * self.discount_factor) + rewards

        # # Compute the Huber Loss
        # ### CODE ####
        # print('step4')
        huber_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # print('good')
        

        # Optimize the model
        self.optimizer.zero_grad()
        huber_loss.backward()
        self.optimizer.step()
