#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
from ddqn import DuelDQN
from collections import deque

# 基本的state shape为(210, 160, 3)
# 为了增加时间序， 我们将3个帧的state拼起来作为state
TIME_STEP_AS_STATE = 3


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
    state = state_buffer[0]
    for i in range(1, TIME_STEP_AS_STATE):
        state = np.concatenate([state, state_buffer[i]], axis=2)
    return state


def do_train():
    env = gym.make("SpaceInvaders-v0")
    state_shape = env.observation_space.shape
    action_count = env.action_space.n
    sess = tf.Session()
    ddqn = DuelDQN(
        sess=sess,
        feature_shape=[None, state_shape[0], state_shape[1], state_shape[2] * TIME_STEP_AS_STATE],
        action_count=action_count,
        memory_size=10000,
        batch_size=128,
        update_network_iter=100,
        choose_e_greedy_increase=0.005
    )
    log_dir = "logs"
    log_writer = tf.summary.FileWriter(log_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    sumary()
    loss_buffer = []
    total_reward_buffer = []
    total_step_buffer = []
    for epoch in range(2000):
        s = env.reset()
        state_buffer = deque(maxlen=TIME_STEP_AS_STATE)
        state_buffer.append(s)
        is_done = False
        cur_state = None
        next_state = None
        total_reward = 0
        total_step = 0
        loss = 0
        while not is_done:
            if cur_state is not None:
                action = ddqn.choose_action(cur_state)
            else:
                action = np.random.randint(0, action_count)
            n_s, reward, is_done, info = env.step(action)
            if is_done:
                reward = -50
            state_buffer.append(n_s)
            if len(state_buffer) >= TIME_STEP_AS_STATE:
                next_state = combine_state(state_buffer)
            if cur_state is not None and next_state is not None:
                ddqn.store(cur_state, action, reward, next_state)
            cur_state = next_state
            if ddqn.data_count % 50 == 0:
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
