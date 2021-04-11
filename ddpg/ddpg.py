import gym
import numpy as np
import itertools
import sys
import matplotlib.pyplot as plt
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal
from copy import deepcopy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Actor(nn.Module):
    # a 2 hidden layer network which takes states as input and produces action within
    # the allowed range 
    def __init__(self, state_dim, act_dim, act_limit):
        super().__init__()
        self.act_limit = act_limit
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.actor = nn.Linear(256, act_dim)
    
    def forward(self, s):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.actor(x)
        x = torch.tanh(x)           # to output in range(-1,1)
        x = self.act_limit * x
        return x

class Critic(nn.Module):
    # a 2 hidden layer network that takes a state and action and produces a
    # Q value as output
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.Q = nn.Linear(256, 1)
    
    def forward(self, s, a):
        x = torch.cat([s,a], dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        q = self.Q(x)
        return torch.squeeze(q, -1)

class ActorCritic(nn.Module):
    # combining the actor and critic into one model
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.state_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]
        
        self.q = Critic(self.state_dim, self.act_dim)       # build Q and policy functions
        self.policy = Actor(self.state_dim, self.act_dim, self.act_limit)
        
    def act(self, state):
        with torch.no_grad():
            return self.policy(state).cpu().numpy()

    def get_action(self, s, noise_scale):
        a = self.act(torch.as_tensor(s, dtype=torch.float32).to(DEVICE))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

class ReplayBuffer:
    def __init__(self, size=1e6):
        self.size = size             # max number of items in buffer
        self.buffer = []             # array to hold buffer
        self.next_id = 0
    
    def __len__(self):
        # returns number of elements in buffer
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done):
        # saves the (s,a,r,s',done) tuple into the buffer
        item = (state, action, reward, next_state, done)
        if len(self.buffer) < self.size:
            self.buffer.append(item)
        else:
            self.buffer[self.next_id] = item
        self.next_id = (self.next_id + 1) % self.size
        
    def sample(self, batch_size=32):
        # returns states, actions, rewards, next_states and done_flags for batch_size random samples
        idxs = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, done_flags = list(zip(*samples))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done_flags)

def q_loss(agent, target_network, states, actions, rewards, next_states, done_flags,
                    gamma=0.99): 
    states = torch.tensor(states, dtype=torch.float).to(DEVICE)            # convert numpy array to torch tensors
    actions = torch.tensor(actions, dtype=torch.float).to(DEVICE)
    rewards = torch.tensor(rewards, dtype=torch.float).to(DEVICE)
    next_states = torch.tensor(next_states, dtype=torch.float).to(DEVICE)
    done_flags = torch.tensor(done_flags.astype('float32'),dtype=torch.float).to(DEVICE)
    
    predicted_qvalues = agent.q(states, actions) # get q-values for all actions in current states using agent network

    with torch.no_grad():                               # bellman backup for Q function
        q__next_state_values = target_network.q(next_states, target_network.policy(next_states))
        target = rewards + gamma * (1 - done_flags) * q__next_state_values

    loss_q = ((predicted_qvalues - target)**2).mean()           # MSE loss against bellman backup
    return loss_q


def policy_loss(agent, states):
    states = torch.tensor(states, dtype=torch.float).to(DEVICE)         # convert numpy array to torch tensors
    predicted_qvalues = agent.q(states, agent.policy(states)).to(DEVICE)
    loss_policy = - predicted_qvalues.mean()
    return loss_policy


def one_step_update(agent, target_network, q_optimizer, policy_optimizer, states, actions, rewards, next_states, done_flags,
                    gamma=0.99, polyak=0.995):
    q_optimizer.zero_grad()     # one step gradient for q-values
    loss_q = q_loss(agent, target_network, states, actions, rewards, next_states, done_flags,
                    gamma)
    loss_q.backward()
    q_optimizer.step()
    
    for params in agent.q.parameters():  # freeze Q-network
        params.requires_grad = False

    policy_optimizer.zero_grad()   # one step gradient for policy network
    loss_policy = policy_loss(agent, states)
    loss_policy.backward()
    policy_optimizer.step()
   
    for params in agent.q.parameters(): # unfreeze Q-network
        params.requires_grad = True
        
    with torch.no_grad():   # update target networks with polyak averaging
        for params, params_target in zip(agent.parameters(), target_network.parameters()):
            params_target.data.mul_(polyak)
            params_target.data.add_((1-polyak)*params.data)

def test_agent(env, agent, num_test_episodes, max_ep_len):
    # test function
    ep_rews, ep_lens = [], []
    for j in range(num_test_episodes):
        state, done, ep_rew, ep_len = env.reset(), False, 0, 0
        while not(done or (ep_len == max_ep_len)):
            state, reward, done, _ = env.step(agent.get_action(state, 0))  # take deterministic actions at test time (noise_scale=0)
            ep_rew += reward
            ep_len += 1
        ep_rews.append(ep_rew)
        ep_lens.append(ep_len)
    return ep_rews, ep_lens, np.mean(ep_rews), np.mean(ep_lens)


def ddpg_train_val(env_fn, seed=0, steps_per_epoch=4000, epochs=10, replay_size=int(1e6), gamma=0.99, polyak=0.995, 
        policy_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, update_after=1000, update_every=50, act_noise=0.1, num_val_episodes=10, max_ep_len=1000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env, test_env = env_fn(), env_fn()
    ep_rews, ep_lens = [], []
    val_ep_rews_arr, val_ep_lens_arr = [], []
    state_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    agent = ActorCritic(env.observation_space, env.action_space).to(DEVICE)
    target_network = deepcopy(agent)
    
    for params in target_network.parameters():       # freeze target networks with respect to optimizers (only update via polyak averaging)
        params.requires_grad = False
    
    replay_buffer = ReplayBuffer(replay_size)        # experience buffer
    
    q_optimizer = optim.Adam(agent.q.parameters(), lr=q_lr)                   # optimizers
    policy_optimizer = optim.Adam(agent.policy.parameters(), lr=policy_lr)
    
    total_steps = steps_per_epoch*epochs
    state, ep_rew, ep_len = env.reset(), 0, 0

    for t in range(total_steps):
        if t > start_steps:
            action = agent.get_action(state, act_noise)
        else:
            action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        ep_rew += reward
        ep_len += 1
        done = False if ep_len==max_ep_len else done  # ignore the 'done' signal if it comes from hitting the time horizon (that is, when it's an artificial terminal 
                                                      # that isn't based on the agent's state)
        replay_buffer.add(state, action, reward, next_state, done)      # store experience to replay buffer
        state = next_state
        if done or (ep_len == max_ep_len):              # end of trajectory handling
            ep_rews.append(ep_rew)
            ep_lens.append(ep_len)
            state, ep_rew, ep_len = env.reset(), 0, 0
        
        if t >= update_after and t % update_every == 0:     # update handling
            for _ in range(update_every):
                states, actions, rewards, next_states, done_flags = replay_buffer.sample(batch_size)
                one_step_update(agent, target_network, q_optimizer, policy_optimizer, 
                        states, actions, rewards, next_states, done_flags,
                        gamma, polyak)
        
        if (t+1) % steps_per_epoch == 0:                # end of epoch handling
            epoch = (t+1) // steps_per_epoch
            val_ep_rews, val_ep_lens, avg_rew, avg_len = test_agent(test_env, agent, num_val_episodes, max_ep_len)
            val_ep_rews_arr.append(val_ep_rews)
            val_ep_lens_arr.append(val_ep_lens)
            print("Epoch: {:.0f}, Training Average Reward: {:.0f}, Training Average Length: {:.0f}, Val Average Reward: {:.0f}, Val Average Length: {:.0f}".format(epoch, np.mean(ep_rews), np.mean(ep_lens), avg_rew, avg_len))
            ep_rews, ep_lens = [], []
    return agent, q_optimizer, policy_optimizer, val_ep_rews_arr, val_ep_lens_arr