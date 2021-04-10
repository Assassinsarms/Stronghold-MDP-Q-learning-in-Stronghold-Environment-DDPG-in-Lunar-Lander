import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
from IPython.display import clear_output
 
class Agent():
    # A class to represent policies the agent can take and evaluation of those policies
    def __init__(self, env, epsilon, lr, episodes, learning_len = 100, discount_factor = 0.999, mode='Train', view=False):
        self.env = env
        self.epsilon = epsilon
        self.lr = lr
        self.episodes = episodes
        self.learning_len = learning_len
        self.discount_factor = discount_factor
        self.mode = mode
        self.action_space = self.env.getActionSpace()
        self.state_space = self.env.getStateSpace()
        print('The number of actions possible are,', self.action_space)
        print('The number of states in the environment are,', self.state_space)
        self.q_table = np.zeros((self.state_space, self.action_space))          
        # initialising state-action values in q-table to 0 
        self.step_max = 1000
        self.view = view

    def greedy_policy(self, current_state):
        return np.argmax(self.q_table[current_state,:])

    def Qlearn(self):
        total_rewards = []                                      
        # holds the total rewards for the evaluation of the q-learning
        # eps_decay = []         # <- uncomment to see the epsilon decay
        for batch in range(self.episodes//self.learning_len):   
            # learning occurs for learning_len number of episodes in a batch 
            for learn in range(self.learning_len):
                current_state = self.env.reset()
                for steps in range(self.step_max):
                    if self.mode == 'Train':
                        if np.random.uniform(0, 1) >= self.epsilon:         
                            #  random choice between exploration (random action) or exploitation (ε-greedy policy)
                            choice_action = self.greedy_policy(current_state)
                        else: choice_action = np.random.randint(0, self.action_space - 1)
                    else:
                        choice_action = self.greedy_policy(current_state)
                    state, reward, done, info = self.env.step(choice_action)
                        # update q_table during training according to the equation
                    if self.mode == 'Train':
                        self.q_table[current_state, choice_action] = self.q_table[current_state, choice_action] + self.lr * \
                            (reward + self.discount_factor * self.q_table[state, self.greedy_policy(state)] - self.q_table[current_state, choice_action])   
                        current_state = state
                    if done == True:
                        break 
                self.epsilon *= 0.99       # ε decay to encourage mainly exploitation at some point
                # eps_decay.append(self.epsilon)   
                # # ^- uncomment to see the epsilon decay
            avg_reward = self.evaluate_policy(self.greedy_policy, self.episodes, self.view) 
            # run evaluation using ε greedy policy according to q_table
            total_rewards.append(avg_reward)
            # if learning_len is 1000 and num episodes is 10000, this holds 10 values which are
            # the average reward per 10000 episodes where 1000 of those episodes were for
            # learning 
        return total_rewards, self.q_table#, eps_decay  # <- uncomment to see the epsilon decay 

    def evaluate_policy(self, policy, episodes, view = False):  
        # evaluates the ε greedy policy after exploration/exploitation/training
        total_evaluation_reward = 0
        for i in range(episodes):
            self.env.reset()
            steps = 0 
            done = False
            state, reward, done, info = self.env.step(policy(self.env.agent_state))
            total_evaluation_reward += (self.discount_factor ** steps) * reward
            steps += 1
            while not done:
                state, reward, done, info = self.env.step(policy(state))
                total_evaluation_reward += (self.discount_factor ** steps) * reward
                steps += 1
                if view == True:
                    print("state", state)
                    self.env.render()
                    time.sleep(0.5)
                    clear_output(wait=True)
        return total_evaluation_reward/episodes

    def save_model(self, filename):
        parameters = {'Q-table': self.q_table, 'Epsilon': self.epsilon, 'Gamma': self.discount_factor,\
            'Learning Rate': self.lr}
        pickle.dump(filename, open(filename+ '.trained', 'wb'), -1)
    
    def load_model(self, filename):
        model = pickle.load(filename, open(filename+'.trained', 'rb'), -1)
        self.q_table, self.epsilon, self.discount_factor, self.lr = model[0], model[1], model[2], model[3]

    def plot_evaluation(self, avg_rewards, learning_len, epsilon, lr, discount_factor):
        episodes = []
        for i in range(len(avg_rewards)):
            episodes.append(i * learning_len + 1)

        fig, ax = plt.subplots(figsize=(10,7)); 
        plt.plot(episodes, avg_rewards)
        plt.title('Stronghold Q-learning Avg. Rewards vs. Episodes')
        plt.xlabel("Episodes")
        plt.ylabel("Avg. Reward per {} Episodes".format(learning_len))
        plt.ylim([np.min(avg_rewards)-50, np.max(avg_rewards)+50])
        plt.legend(['Eps={}, Lr={}, Gamma={}'.format(epsilon, lr, discount_factor)], loc='lower right')
        plt.show();

    def plot_policy(self, q_table):
        best_value = np.max(q_table, axis = 1)[:-1].reshape((self.env.size, self.env.size))
        best_action = np.argmax(q_table, axis = 1)[:-1].reshape((self.env.size, self.env.size))
        
        print("Best Policy")
        fig, ax = plt.subplots(figsize=(10,7))
        im = ax.imshow(best_value, cmap ='Set3')

        for i in range(best_value.shape[0]):
            for j in range(best_value.shape[1]):
                if self.env.land[i, j] == 1 or self.env.land[i, j] == 3 or self.env.land[i, j] == 5:
                    sign = self.env.dict_map_display[self.env.land[i,j]]
                elif best_action[i, j] == 0:
                    sign = '^'
                elif best_action[i, j] == 1:
                    sign = 'v'
                elif best_action[i, j] == 2:
                    sign = '<'
                elif best_action[i, j] == 3:
                    sign = '>'
                text = ax.text(j, i, sign, ha = "center", va = "center", color = "black", fontsize='x-large')
                
        cbar = ax.figure.colorbar(im, ax = ax, fraction= 0.046, pad = 0.04)
        fig.tight_layout()
        plt.show(); 

    def print_q_table(self, q_table):
        print("Some Q-table values")
        df = pd.DataFrame(q_table)
        df.columns = ['1', '2', '3', '4']
        print(df[(df > 0).any(1)])