#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
# 由于gpu资源有限， 可以通过该选项设置agent使用cpu模式
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import numpy as np
import random
import gym
from dqn import DQN
from collections import deque
import time
import cv2
from config import game_config


SAME_ACTION_STEP = game_config["SAME_ACTION_STEP"]
TIME_STEP_AS_STATE = game_config["TIME_STEP_AS_STATE"]
IMAGE_HEIGHT = game_config["IMAGE_HEIGHT"]
IMAGE_WIDTH = game_config["IMAGE_WIDTH"]
WEIGHT_DATA_PATH = game_config["WEIGHT_DATA_PATH"]


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
    s = s[25: -12, :, :]
    s = cv2.resize(cv2.cvtColor(s, cv2.COLOR_RGB2GRAY), (IMAGE_HEIGHT, IMAGE_WIDTH))
    return np.array(s, dtype=np.uint8)


def boot_game():
    env = gym.make(game_config["GAME_NAME"])
    action_count = env.action_space.n - 1
    sess = tf.Session()
    dqn = DQN(
        sess=sess,
        feature_shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, TIME_STEP_AS_STATE],
        action_count=action_count,
        choose_e_greedy=0.95,
        choose_e_greedy_increase=None,
        is_duel=game_config["IS_DUEL"]
    )
    sess.run(tf.global_variables_initializer())
    dqn.load_weights(WEIGHT_DATA_PATH)
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
        if cur_state is not None:
            action = dqn.choose_action(cur_state)
        else:
            action = np.random.randint(0, action_count)
        reward = 0
        for f in range(SAME_ACTION_STEP):
            n_s, _reward, _is_done, info = env.step(action + 1)
            env.render()
            time.sleep(0.02)
            n_s = transfer_observation(n_s)
            state_buffer.append(n_s)
            if _is_done:
                is_done = True
            reward += _reward
        if len(state_buffer) >= TIME_STEP_AS_STATE:
            next_state = combine_state(state_buffer)
        cur_state = next_state
        total_step += 1
        total_reward += reward
    print("game over. total_step: ", total_step, "total_reward: ", total_reward)


if __name__ == '__main__':
    boot_game()
