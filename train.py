#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import random
import gym
from dqn import DQN
from collections import deque
import cv2
import os
from config import game_config, train_config

# 基本的state shape为(210, 160, 3)
# 为了增加时间序， 我们将3个帧的state拼起来作为state
SAME_ACTION_STEP = game_config["SAME_ACTION_STEP"]
TIME_STEP_AS_STATE = game_config["TIME_STEP_AS_STATE"]
IMAGE_HEIGHT = game_config["IMAGE_HEIGHT"]
IMAGE_WIDTH = game_config["IMAGE_WIDTH"]
WEIGHT_DATA_PATH = game_config["WEIGHT_DATA_PATH"]


def set_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def sumary():
    total_variables = 0
    for var in tf.trainable_variables():
        shape = []
        var_cnt = 1
        for dim in var.get_shape():
            var_cnt *= dim.value
            shape.append(dim.value)
        total_variables += var_cnt
        print("var_name: ", var.name)
        print("shape: ", shape,
              "variables: ", var_cnt)
    print("total_variables: ", total_variables)


def combine_state(state_buffer):
    state = []
    for i in range(0, TIME_STEP_AS_STATE):
        state.append(state_buffer[i])
    return np.array(state).transpose((2, 1, 0))


def transfer_observation(s):
    s = s[25: -12, :, :]
    s = cv2.resize(cv2.cvtColor(s, cv2.COLOR_RGB2GRAY), (IMAGE_HEIGHT, IMAGE_WIDTH))
    return np.array(s, dtype=np.uint8)


def do_train():
    if not os.path.exists(WEIGHT_DATA_PATH):
        os.makedirs(os.path.dirname(WEIGHT_DATA_PATH))
    env = gym.make(game_config["GAME_NAME"])
    action_count = env.action_space.n - 1
    sess = tf.Session()
    dqn = DQN(
        sess=sess,
        feature_shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, TIME_STEP_AS_STATE],
        action_count=action_count,
        memory_size=train_config["MEMORY_SIZE"],
        batch_size=train_config["BATCH_SIZE"],
        update_network_iter=train_config["UPDATE_TARGET_ITER"],
        choose_e_greedy_increase=train_config["CHOOSE_E_GREEDY_INCREASEMENT"],
        learning_rate=train_config["LEARNING_RATE"],
        is_duel=game_config["IS_DUEL"]
    )
    log_dir = game_config["LOG_PATH"]
    log_writer = tf.summary.FileWriter(log_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    try:
        dqn.load_weights(WEIGHT_DATA_PATH)
        print("load weights from [%s] success." % WEIGHT_DATA_PATH)
    except Exception as e:
        print("load weights from [%s] failed: %s" % (WEIGHT_DATA_PATH, e.message))
        print("init ddqn as random weights.")
    sumary()
    for epoch in range(50000):
        s = env.reset()
        s = transfer_observation(s)
        state_buffer = deque(maxlen=TIME_STEP_AS_STATE)
        state_buffer.append(s)
        is_done = False
        cur_state = None
        next_state = None
        total_reward = 0
        total_step = 0
        loss_sum = 0
        while not is_done:
            if cur_state is not None:
                action = dqn.choose_action(cur_state)
            else:
                action = np.random.randint(0, action_count)
            reward = 0
            for f in range(SAME_ACTION_STEP):
                n_s, _reward, _is_done, info = env.step(action + 1)
                n_s = transfer_observation(n_s)
                state_buffer.append(n_s)
                if _is_done:
                    is_done = True
                reward += _reward
            if len(state_buffer) >= TIME_STEP_AS_STATE:
                next_state = combine_state(state_buffer)
            if cur_state is not None and next_state is not None:
                dqn.store(cur_state, action, reward, next_state, is_done)
            cur_state = next_state
            loss = dqn.learn(log_writer)
            if loss is not None:
                loss_sum = loss_sum + loss
            # print("step: ", total_step, "reward: ", reward, "loss: ", loss)
            total_step += 1
            total_reward += reward
        print("epoch: ", epoch,
              "total_step: ", total_step,
              "total_reward: ", total_reward,
              "loss: ", loss_sum / total_step
              )
        dqn.add_game_total_reward(total_reward)
        if epoch % 20 == 0:
            dqn.save_weight(WEIGHT_DATA_PATH)
    log_writer.close()


if __name__ == '__main__':
    do_train()
