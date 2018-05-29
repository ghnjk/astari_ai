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
import time
from config import game_config, train_config, run_mode


# 基本的state shape为(210, 160, 3)
# 为了增加时间序， 我们将3个帧的state拼起来作为state
SAME_ACTION_STEP = game_config["same_action_step"]
TIME_STEP_AS_STATE = game_config["time_step_as_state"]
IMAGE_HEIGHT = game_config["image_height"]
IMAGE_WIDTH = game_config["image_width"]
WEIGHT_DATA_PATH = game_config["weight_data_path"]
GAME_NAME = game_config["game_name"]
IS_DUEL = game_config["is_duel"]
LOG_PATH = game_config["log_path"]


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
    if not os.path.isdir(WEIGHT_DATA_PATH):
        os.makedirs(WEIGHT_DATA_PATH)
    model_weight = os.path.join(WEIGHT_DATA_PATH, "model_weight.ckpt")
    env = gym.make(GAME_NAME)
    if GAME_NAME == "Breakout-v0":
        action_count = env.action_space.n - 1
    else:
        action_count = env.action_space.n
    sess = tf.Session()
    dqn = DQN(
        sess=sess,
        feature_shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, TIME_STEP_AS_STATE],
        action_count=action_count,
        memory_size=train_config["relay_memory_size"],
        batch_size=train_config["batch_size"],
        update_network_iter=train_config["update_target_net_per_iter"],
        choose_e_greedy_increase=train_config["choose_e_greedy_increase"],
        learning_rate=train_config["learning_rate"],
        is_duel=IS_DUEL
    )
    log_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
    sess.run(tf.global_variables_initializer())
    try:
        dqn.load_weights(model_weight)
        print("load weights from [%s] success." % model_weight)
    except Exception as e:
        print("load weights from [%s] failed: %s" % (model_weight, e.message))
        print("init ddqn as random weights.")
    sumary()
    for epoch in range(train_config["train_epoch"]):
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
            if GAME_NAME == "Breakout-v0":
                real_act = action + 1
            else:
                real_act = action
            reward = 0
            for f in range(SAME_ACTION_STEP):
                n_s, _reward, _is_done, info = env.step(real_act)
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
        if epoch % 500 == 0:
            dqn.save_weight(model_weight)
            backup = os.path.join("data/backup", str(epoch))
            if not os.path.exists(backup):
                os.makedirs(backup)
            backup = os.path.join(backup, "model_weight.ckpt")
            dqn.save_weight(backup)
    log_writer.close()


def boot_game():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    model_weight = os.path.join(WEIGHT_DATA_PATH, "model_weight.ckpt")
    env = gym.make(GAME_NAME)
    if GAME_NAME == "Breakout-v0":
        action_count = env.action_space.n - 1
    else:
        action_count = env.action_space.n
    sess = tf.Session()
    dqn = DQN(
        sess=sess,
        feature_shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, TIME_STEP_AS_STATE],
        action_count=action_count,
        choose_e_greedy=0.95,
        choose_e_greedy_increase=None,
        is_duel=IS_DUEL
    )
    sess.run(tf.global_variables_initializer())
    dqn.load_weights(model_weight)
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
        if GAME_NAME == "Breakout-v0":
            real_act = action + 1
        else:
            real_act = action
        reward = 0
        for f in range(SAME_ACTION_STEP):
            n_s, _reward, _is_done, info = env.step(real_act)
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
    if run_mode == "train":
        do_train()
    else:
        boot_game()
