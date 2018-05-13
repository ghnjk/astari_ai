#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import random
import gym
from ddqn import DuelDQN
from collections import deque
import time
import cv2


# 基本的state shape为(210, 160, 3)
# 为了增加时间序， 我们将3个帧的state拼起来作为state
TIME_STEP_AS_STATE = 3
WEIGHT_DATA_PATH = "data/model_weights/ddqn_weights.ckpt"


def set_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def combine_state(state_buffer):
    state = []
    for i in range(0, TIME_STEP_AS_STATE):
        state.append(state_buffer[i])
    return np.array(state).transpose((2, 1, 0))


def transfer_observation(s):
    return cv2.resize(cv2.cvtColor(s, cv2.COLOR_RGB2GRAY), (s.shape[0] / 2, s.shape[1] / 2))


def boot_game():
    env = gym.make("SpaceInvaders-v0")
    state_shape = env.observation_space.shape
    action_count = env.action_space.n
    sess = tf.Session()
    ddqn = DuelDQN(
        sess=sess,
        feature_shape=[None, state_shape[0] / 2, state_shape[1] / 2, TIME_STEP_AS_STATE],
        action_count=action_count,
        memory_size=5000,
        batch_size=256,
        update_network_iter=100,
        choose_e_greedy=1.0,
        choose_e_greedy_increase=None
    )
    sess.run(tf.global_variables_initializer())
    ddqn.load_weights(WEIGHT_DATA_PATH)
    is_done = False
    s = env.reset()
    s = transfer_observation(s)
    state_buffer = deque(maxlen=TIME_STEP_AS_STATE)
    state_buffer.append(s)
    cur_state = None
    next_state = None
    total_reward = 0
    total_step = 0
    while not is_done:
        env.render()
        if cur_state is not None:
            action = ddqn.choose_action(cur_state)
        else:
            action = np.random.randint(0, action_count)
        n_s, reward, is_done, info = env.step(action)
        n_s = transfer_observation(n_s)
        state_buffer.append(n_s)
        if len(state_buffer) >= TIME_STEP_AS_STATE:
            next_state = combine_state(state_buffer)
        cur_state = next_state
        total_step += 1
        total_reward += reward
        time.sleep(0.2)
    print("game over. total_step: ", total_step, "total_reward: ", total_reward)


if __name__ == '__main__':
    boot_game()
