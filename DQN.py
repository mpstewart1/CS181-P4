from __future__ import division
import numpy as np
import numpy.random as npr
import sys
import math
import random
from SwingyMonkey import SwingyMonkey
import math
import pickle
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import pygame as pg

class DQN(object):

    def __init__(self):
        self.state_size = 6
        self.action_size = 2

        #Create a replay buffer using deque.
        self.memory = deque(maxlen=2000)
        self.steps = 0
        self.epoch = 0

        # hyperparameters
        self.alpha = 0.8
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.update_freq = 100
        self.start_len_memory = 500
        self.n_mini_batch = 50

        # state parameters
        self.last_state  = None
        self.last_action = None
        self.last_reward = None


        # neural net hyperparameters
        self.lr = 0.001
        self.nnodes = 128

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.nnodes, activation='relu', input_dim=self.state_size))
        model.add(Dense(self.nnodes, activation='relu'))
        # output layer
        model.add(Dense(self.action_size, activation='linear'))

        optimizer = Adam(lr=self.lr)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def update_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        #self.iter += 1
        self.epoch+=1

    def update_memory(self, state):
        # don't save initial start state
        if self.last_reward is None:
            return
        # current state
        state_0 = self.last_state
        # action taken
        action = self.last_action
        # reward of action
        reward = self.last_reward
        # next state
        state_1 = np.array([state['tree']['dist'],
                            state['tree']['top'],
                            state['tree']['bot'],
                            state['monkey']['vel'],
                            state['monkey']['top'],
                            state['monkey']['bot']]).reshape(-1,self.state_size)
        # stopping condition
        done = self.last_reward < 0.0

        # collate states
        self.memory.append([state_0, action, reward, state_1, done])
        # update target model
        if self.steps % self.update_freq == 0:
            self.update_model()
        self.steps+=1

    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        # epsilon-greedy policy

        #random action = generate a random action by randomly sample a number from 0 to 1
        #With probability epsion, select the random action, and
        #with probability 1-epsion select a greedy action (choose the max in the Q table)

        # save to memory before taking any new action
        self.update_memory(state)
        state_1 = np.array([state['tree']['dist'],
                            state['tree']['top'],
                            state['tree']['bot'],
                            state['monkey']['vel'],
                            state['monkey']['top'],
                            state['monkey']['bot']]).reshape(-1,self.state_size)
        # epsilon greedy
        if np.random.binomial(1,self.epsilon):
            new_action = random.randrange(self.action_size)
        else:
            # or should i train it regardless
            self.train_model()
            new_action = np.argmax(self.model.predict(state_1)[0]) # edit subsequently
        self.last_state  = state_1
        self.last_action = new_action

        # start with more exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return self.last_action

    def train_model(self):
        # train model once you have sufficient memory
        if len(self.memory) < self.start_len_memory:
            return
        # random mini batch of transitions from memory
        mini_batch = random.sample(self.memory, self.n_mini_batch)

        state_batch = []
        q_batch = []
        for state_0, action, reward, state_1, done in mini_batch:
            q_new = reward
            if not done:
                # main difference between ddqn and dqn, where we select the action with q model
                # and evaluate with target model
                q_new = reward + self.gamma*np.max(self.target_model.predict(state_1)[0])
            q = self.model.predict(state_0)
            q[0][action] = (1-self.alpha)*q[0][action]+self.alpha*q_new

            state_batch.append(state_0[0])
            q_batch.append(q[0])

        # reshape
        state_batch = np.array(state_batch).reshape(-1,self.state_size)
        q_batch = np.array(q_batch).reshape(-1,self.action_size)
        self.model.fit(state_batch, q_batch, epochs=1, verbose=0)

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward

def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)
        if len(hist) < 100:
            avgscore = np.mean(hist)
        else:
            avgscore = np.mean(hist[-100:])
        print("epoch:",ii, "highest:", np.max(hist),
            "current score:", swing.score, "average:", avgscore)
        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return

if __name__ == '__main__':

	# Select agent.
	agent = DQN()

	# Empty list to save history.
	hist = []

	# Run games.
	run_games(agent, hist, 1000, 1)

	# Save history.
	np.save('hist_dqn',np.array(hist))
