#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import random
import gym
import matplotlib.pyplot as plt
import time


def set_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def plot_image(state):
    plt.imshow(state)
    plt.show()



def boot_game():
    env = gym.make("SpaceInvaders-v0")
    print("observation_space: ", env.observation_space.shape)
    print("action_space: ", env.action_space)
    cur_state = env.reset()
    is_done = False
    reward_list = []
    while not is_done:
        env.render()
        action = env.action_space.sample()
        print("action: ", action)
        next_state, reward, is_done, info = env.step(action)
        reward_list.append(reward)
        print("reward: ", reward)
        cur_state = next_state
        time.sleep(1)
    plot_image((cur_state))
    plt.plot(reward_list)
    plt.show()


if __name__ == '__main__':
    boot_game()
