#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
from ddqn import DuelDQN
from collections import deque
import cv2


# 基本的state shape为(210, 160, 3)
# 为了增加时间序， 我们将3个帧的state拼起来作为state
TIME_STEP_AS_STATE = 3
WEIGHT_DATA_PATH = "data/model_weights/ddqn_weights.ckpt"


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
    return cv2.resize(cv2.cvtColor(s, cv2.COLOR_RGB2GRAY), (s.shape[0] / 2, s.shape[1] / 2))


def evaluate_last_score(total_step, total_reward):
    if total_reward > 700:
        score = 1000
    elif total_reward > 600:
        score = 800
    elif total_reward > 500 and total_step > 1500:
        score = 800
    elif total_reward > 500 and total_step > 1000:
        score = 200
    elif total_reward > 300 and total_step > 800:
        score = 0
    elif total_reward > 200 and total_step > 1000:
        score = -400
    elif total_reward > 100 and total_step > 600:
        score = -800
    else:
        score = -1000
    score += total_reward / 2
    score /= 10
    print("evaluate_last_score: ", score)
    return score


def do_train():
    env = gym.make("SpaceInvaders-v0")
    state_shape = env.observation_space.shape
    action_count = env.action_space.n
    sess = tf.Session()
    ddqn = DuelDQN(
        sess=sess,
        feature_shape=[None, state_shape[0] / 2, state_shape[1] / 2, TIME_STEP_AS_STATE],
        action_count=action_count,
        memory_size=10000,
        batch_size=256,
        update_network_iter=100,
        choose_e_greedy_increase=0.005,
        learning_rate=0.00025
    )
    log_dir = "logs"
    log_writer = tf.summary.FileWriter(log_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    try:
        ddqn.load_weights(WEIGHT_DATA_PATH)
        print("load weights from [%s] success." % WEIGHT_DATA_PATH)
    except Exception as e:
        print("load weights from [%s] failed: %s" % (WEIGHT_DATA_PATH, e.message))
        print("init ddqn as random weights.")
    sumary()
    loss_buffer = []
    total_reward_buffer = []
    total_step_buffer = []
    for epoch in range(2000):
        s = env.reset()
        s = transfer_observation(s)
        state_buffer = deque(maxlen=TIME_STEP_AS_STATE)
        state_buffer.append(s)
        is_done = False
        cur_state = None
        total_reward = 0
        total_step = 0
        loss = 0
        while not is_done:
            if cur_state is not None:
                action = ddqn.choose_action(cur_state)
            else:
                action = np.random.randint(0, action_count)
            reward = 0
            for f in range(TIME_STEP_AS_STATE):
                n_s, _reward, _is_done, info = env.step(action)
                n_s = transfer_observation(n_s)
                state_buffer.append(n_s)
                if _is_done:
                    is_done = True
                reward += _reward
            next_state = combine_state(state_buffer)
            if cur_state is not None and next_state is not None:
                ddqn.store(cur_state, action, reward, next_state, is_done)
            cur_state = next_state
            loss = ddqn.learn()
            # print("step: ", total_step, "reward: ", reward, "loss: ", loss)
            loss_buffer.append(loss)
            total_step += 1
            total_reward += reward
        print("epoch: ", epoch,
              "total_step: ", total_step,
              "total_reward: ", total_reward,
              "loss: ", loss
              )
        total_reward_buffer.append(total_reward)
        total_step_buffer.append(total_step)
        ddqn.save_weight(WEIGHT_DATA_PATH)
    log_writer.close()
    plt.subplot(221)
    plt.plot(total_step_buffer)
    plt.xlabel("epoch")
    plt.ylabel("step_count")
    plt.subplot(222)
    plt.plot(total_reward_buffer)
    plt.xlabel("epoch")
    plt.ylabel("reward")
    plt.subplot(212)
    plt.plot(loss_buffer)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()


if __name__ == '__main__':
    do_train()
