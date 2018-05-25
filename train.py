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
SAME_ACTION_STEP = 1
TIME_STEP_AS_STATE = 4
IMAGE_HEIGHT = 84
IMAGE_WIDTH = 84
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
    s = s[25: -12, :, :]
    s = cv2.resize(cv2.cvtColor(s, cv2.COLOR_RGB2GRAY), (IMAGE_HEIGHT, IMAGE_WIDTH))
    s = s / 256.0
    return s


def do_train():
    env = gym.make("Breakout-v0")
    state_shape = env.observation_space.shape
    action_count = env.action_space.n - 1
    sess = tf.Session()
    ddqn = DuelDQN(
        sess=sess,
        feature_shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, TIME_STEP_AS_STATE],
        action_count=action_count,
        memory_size=30000,
        batch_size=32,
        update_network_iter=5000,
        choose_e_greedy_increase=0.005,
        learning_rate=0.001
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
    for epoch in range(5000):
        s = env.reset()
        s = transfer_observation(s)
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
                ddqn.store(cur_state, action, reward, next_state, is_done)
            cur_state = next_state
            if total_step % 1 == 0:
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
        if epoch % 20 == 0:
            print("epoch ", epoch, " last average rewards: ",
                  np.average(np.array(total_reward_buffer)))
            total_reward_buffer = []
            total_step_buffer = []
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
