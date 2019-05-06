# CS181-P4 - Reinforcement learning with Swingy Monkey

This repository documents the work completed by **Matthew Stewart**, **Claire Stolz**, and **Shane Ong** for the final practical of CS181: Machine Learning, involving the application of reinforcement learning to the flappy bird-style game 'Swingy Monkey'.

In this documentation we outline the methods used to attain the highest possible score on the game by using Q-learning and its deep learning-based extensions, commonly referred to as DQN (Deep Q-Learning) and DDQN (Double Deep Q-Learning).

<p align="center">
  <img width="700" height="500" src="https://github.com/mrdragonbear/CS181-P4/blob/master/Swingy_Monkey.png">
</p>

## Installing Swingy Monkey

To install Swingy Monkey, follow the instructions below. For tips on installing `pygame`, see the [pygame](http://stackoverflow.com/questions/22314904/installing-pygame-with-enthought-canopy-on-mac) documentation.

>     pip install pygame

## Running Models

To run the individual models, first make sure you have `pygame` installed. Then use the commands for each of the individual models:

>     python Q-learning_model.py 
>     python DQN.py
>     python DDQN.py

## Overview of Models

The score for the three models developed in this practical are outlined in the table below.
  
Model | Maximum Score | Mean Score
:------------: | :-------------: | :-------------:
**Q-Learning** | 10 | 4.1
**DQN** | 169 | 22.7
**DDQN** | 118 | 17.6

Hyperparameter optimization was performed in order to optimize for the &epsilon;-greedy policy, the learning rate &alpha;, as well as the discount rate &gamma;, and other factors such as the neural architecture and hyperparameters of the deep neural network in the deep learning models.
