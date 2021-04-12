import copy
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

import torch
from torch import cuda
from torch import device
from torch import from_numpy
from torch import nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, dataloader


if cuda.is_available():
    device = device("cuda")
else:
    device = device("cpu")

"""
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
"""

class State:
    def __init__(self, price, past_price_0=0, past_price_1=0, past_price_2=0, hold=0, hold_price=0, buy=0):
        self.p0 = past_price_0
        self.p1 = past_price_1
        self.p2 = past_price_2
        self.price = price
        self.hold = hold
        self.hold_price = hold_price
        self.buy = buy

    def update(self, price, action, hold_price, buy):
        self.p2 = self.p1
        self.p1 = self.p0
        self.p0 = self.price
        self.price = price
        self.hold += action
        self.hold_price = hold_price
        self.buy = buy
        
    
    def action_sample(self):
        if self.hold == 1:
            return random.randint(0,2)-1
        elif self.hold == 0:
            return random.randint(0,3)-1
        else:
            return random.randint(0,2)

class Preprocess:
    def __init__(self, df):
        self.arr = np.array(df)
        self.new = []
    def past(self):
        for i in range(len(self.arr)):
            self.new.append(self.arr[i][0])
        return self.new

class Environment:
    def __init__(self, data):
        self.li = data
           
    def render(self):
        start_point = random.randint(0,len(self.li)-20)
        self.t = start_point
        self.trajectory = self.li[start_point:start_point+20]
        self.state = State(self.trajectory[0])
        return self.trajectory

    def step(self, action, now_state):
        self.t += 1
        new_state = copy.deepcopy(now_state)
        if now_state.buy == 1:
            if action == 0:
                new_state.update(price = self.li[self.t], action = action, hold_price=self.li[self.t-1], buy=0)
                reward = 0-now_state.past_price_0
            elif action == 1:
                if now_state.hold == 1:
                    new_state.update(price = self.li[self.t], action = action, hold_price=self.li[self.t-1], buy=1)
                    reward = -99999
                elif now_state.hold == 0:
                    new_state.update(price = self.li[self.t], action = action, hold_price=self.li[self.t-1], buy=1)
                    reward = 0-now_state.past_price_0
                
                # else:
                #     reward = 0-now_state.past_price_0
                #     now_state.hold += action
            else: #action == -1
                if now_state.hold == 1:
                    new_state.update(price = self.li[self.t], action = action, hold_price=0, buy=0)
                    reward = now_state.price-now_state.past_price_0
                elif now_state.hold == 0:
                    new_state.update(price = self.li[self.t], action = action, hold_price=self.li[self.t-1], buy=0)
                    reward = now_state.price-now_state.past_price_0

        elif now_state.buy == 0:
            if action == 0:
                new_state.update(price = self.li[self.t], action = action, hold_price=0, buy=0)
                reward = 0
            elif action == 1:
                if now_state.hold == 1:
                    new_state.update(price = self.li[self.t], action = action, hold_price=self.li[self.t-1], buy=1)
                    reward = -99999
                elif now_state.hold == 0:
                    new_state.update(price = self.li[self.t], action = action, hold_price=self.li[self.t], buy=1)
                    reward = 0  
                else:
                    new_state.update(price = self.li[self.t], action = action, hold_price=self.li[self.t], buy=1)
                    reward = 0
                                    
            else:
                if now_state.hold == 1:
                    new_state.update(price = self.li[self.t], action = action, hold_price=0, buy=0)
                    reward = now_state.price-now_state.hold_price
                elif now_state.hold == 0:
                    new_state.update(price = self.li[self.t], action = action, hold_price=self.li[self.t], buy=0)
                    reward = now_state.price
                else:
                    new_state.update(price = self.li[self.t], action = action, hold_price=self.li[self.t], buy=0)
                    reward = -99999
        
        return reward, new_state

# import pandas as pd

# df = pd.read_csv("training.csv", header=None)
# new = Preprocess(df).past()

# print(State(Environment(new).render()[0]).action_sample())

   
class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()

        # 輸入層 (state) 到隱藏層，隱藏層到輸出層 (action)
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x) # ReLU activation
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden):
        self.eval_net, self.target_net = Net(n_states, n_actions, n_hidden), Net(n_states, n_actions, n_hidden)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters())
        self.loss_func = nn.MSELoss()
        

    def choose_action(self, x):
        x = torch.unsqueeze(torch.Tensor(x), 0)
        # input only one sample
        if np.random.uniform() < epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            
        else:   # random
            action = env.state.action_sample()
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(memory_capacity, batch_size)
        b_memory = self.memory[sample_index, :]
        b_state = torch.FloatTensor(b_memory[:, :self.n_states])
        b_action = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))
        b_reward = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
        b_next_state = torch.FloatTensor(b_memory[:, -self.n_states:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_state).gather(1, b_action)  # shape (batch, 1)
        q_next = self.target_net(b_next_state).detach()     # detach from graph, don't backpropagate
        q_target = b_reward + gamma * q_next.max(1)[0].view(batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Hyper parameters
n_hidden = 50
batch_size = 32
lr = 0.01                 # learning rate
epsilon = 0.1             # epsilon-greedy
gamma = 0.9               # reward discount factor
target_replace_iter = 100 # target network 更新間隔
memory_capacity = 2000
n_episodes = 4000

import pandas as pd

df = pd.read_csv("training.csv", header=None)
new = Preprocess(df).past()

print(State(Environment(new).render()[0]).action_sample())

env = Environment(new)
n_states = len(env.render())
n_actions = 0
dqn = DQN(n_states, n_actions, n_hidden)

print('\nCollecting experience...')
for i_episode in range(400):
    reward_t = 0
    trajectory = env.render()
    for i in range(len(trajectory)):
        state = State(trajectory[i])
        action = dqn.choose_action(state)

        # take action
        reward, new_state = env.step(action, state)
        dqn.store_transition(state, action, reward, new_state)

        reward_t += reward
        
        

        if dqn.memory_counter > memory_capacity:
            dqn.learn()
            print('Ep: ', i_episode,
                '| Ep_r: ', round(reward_t, 2))

