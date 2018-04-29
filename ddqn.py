#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from memory import RandomMemory


class DuelDQN(object):

    def __init__(self, sess, feature_shape, action_count,
                 memory_size=10000,
                 batch_size=128,
                 update_network_iter=200,
                 learning_rate=0.001,
                 reward_decay=0.95,
                 choose_e_greedy=0.9,
                 choose_e_greedy_increase=None):
        self.sess = sess
        self.feature_shape = feature_shape
        self.action_count = action_count
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.choose_e_greedy = choose_e_greedy
        self.choose_e_greedy_increase = choose_e_greedy_increase
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = RandomMemory(memory_size=self.memory_size)
        self.update_network_iter = update_network_iter
        if self.choose_e_greedy_increase is None:
            self.choose_random_rate = self.choose_e_greedy
        else:
            self.choose_random_rate = 0
        self.data_count = 0
        self.learn_counter = 0

        # tf inputs
        self.tf_cur_state = None
        self.tf_q_target = None
        self.tf_next_state = None
        # tf outputs
        self.tf_q_eval = None
        self.tf_q_real = None
        self.tf_loss = None
        # tf variables
        self.tf_eval_net_variables = None
        self.tf_target_net_variables = None
        # tf op
        self.tf_train_op = None
        self.tf_update_network_o = None

        self._build_net()

    def choose_action(self, cur_state):
        cur_state = np.array(cur_state)[np.newaxis, :]
        if np.random.uniform() < self.choose_random_rate:
            act_values = self.sess.run(
                self.tf_q_eval,
                feed_dict={
                    self.tf_cur_state: cur_state
                }
            )
            return np.argmax(act_values)
        else:
            return np.random.randint(0, self.action_count)

    def store(self, cur_state, action, reward, next_state):
        item = [cur_state, action, reward, next_state]
        self.memory.append(item)
        self.data_count += 1

    def learn(self):
        if self.data_count < self.batch_size:
            return None

        if self.learn_counter % self.update_network_iter == 0:
            self.sess.run(self.tf_update_network_o)
            print("update target network")

        samples = self.memory.choose_sample(sample_count=self.batch_size)
        cur_state = []
        action = []
        reward = []
        next_state = []
        for item in samples:
            cur_state.append(item[0])
            action.append(item[1])
            reward.append(item[2])
            next_state.append(item[3])

        q_eval, q_real = self.sess.run(
            [self.tf_q_eval, self.tf_q_real],
            feed_dict={
                self.tf_cur_state: cur_state,
                self.tf_next_state: next_state
            }
        )
        # 计算q_target = reward + reward_decay * max(q_real)
        q_target = q_eval.copy()
        batch_idx = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_idx, action] = reward + self.reward_decay * np.max(q_real, axis=1)

        _, loss = self.sess.run(
            [self.tf_train_op, self.tf_loss],
            feed_dict={
                self.tf_cur_state: cur_state,
                self.tf_q_target: q_target
            }
        )

        if self.choose_e_greedy_increase is not None:
            self.choose_random_rate += self.choose_e_greedy_increase
            if self.choose_random_rate > self.choose_e_greedy:
                self.choose_random_rate = self.choose_e_greedy
        self.learn_counter += 1

        return loss

    def _build_net(self):
        with tf.name_scope("inputs"):
            self.tf_cur_state = tf.placeholder(tf.float32, self.feature_shape, name="cur_state")
            self.tf_next_state = tf.placeholder(tf.float32, self.feature_shape, name="next_state")
            self.tf_q_target = tf.placeholder(tf.float32, [None, self.action_count], name="q_target")

        self.tf_q_eval = self._build_q_net(self.tf_cur_state, "eval_net")
        self.tf_q_real = self._build_q_net(self.tf_next_state, "target_net")
        with tf.variable_scope("loss"):
            self.tf_loss = tf.reduce_mean(tf.squared_difference(self.tf_q_target, self.tf_q_eval))
        with tf.variable_scope("train"):
            self.tf_train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.tf_loss)

        self.tf_eval_net_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval_net")
        self.tf_target_net_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")
        with tf.variable_scope("update_network"):
            self.tf_update_network_o = [
                tf.assign(t, e) for t, e in zip(self.tf_target_net_variables, self.tf_eval_net_variables)
            ]

    def _build_q_net(self, inputs, scope):
        w_initializer = tf.random_normal_initializer(0, 3)
        bias_initializer = tf.constant_initializer(0.1)
        with tf.variable_scope(scope):
            with tf.variable_scope("image_feature"):
                with tf.variable_scope("conv_layer_1"):
                    conv_layer_1 = self._add_one_feature_layer(inputs, conv2d_filter=8, dropout_rate=0.3)

                # with tf.variable_scope("conv_layer_2"):
                #     conv_layer_2 = self._add_one_feature_layer(
                # inputs=conv_layer_1, conv2d_filter=16, dropout_rate=0.3)

                with tf.variable_scope("dense_layer_1"):
                    flattten_layer = tf.layers.flatten(
                        inputs=conv_layer_1,
                        name="flatter_layer"
                    )
                    dense_layer = tf.layers.dense(
                        inputs=flattten_layer,
                        units=64,
                        activation=tf.nn.tanh,
                        kernel_initializer=w_initializer,
                        bias_initializer=bias_initializer,
                        name="dense_layer"
                    )
                tf_img_feature = dense_layer
            with tf.variable_scope("state_v"):
                # dense_layer = tf.layers.dense(
                #     inputs=tf_img_feature,
                #     units=32,
                #     activation=tf.nn.relu,
                #     kernel_initializer=w_initializer,
                #     bias_initializer=bias_initializer,
                #     name="dense_layer"
                # )
                state_v = tf.layers.dense(
                    inputs=tf_img_feature,
                    units=1,
                    activation=None,
                    kernel_initializer=w_initializer,
                    bias_initializer=bias_initializer,
                    name="state_v"
                )
            with tf.variable_scope("action_adavantage"):
                # dense_layer = tf.layers.dense(
                #     inputs=tf_img_feature,
                #     units=32,
                #     activation=tf.nn.relu,
                #     kernel_initializer=w_initializer,
                #     bias_initializer=bias_initializer,
                #     name="dense_layer"
                # )
                action_adavantage = tf.layers.dense(
                    inputs=tf_img_feature,
                    units=self.action_count,
                    activation=None,
                    kernel_initializer=w_initializer,
                    bias_initializer=bias_initializer,
                    name="action_adavantage"
                )
            q = state_v + (action_adavantage - tf.reduce_mean(action_adavantage, axis=1, keepdims=True))
        return q

    @staticmethod
    def _add_one_feature_layer(inputs, conv2d_filter, dropout_rate):
        w_initializer = tf.random_normal_initializer(0, 3)
        bias_initializer = tf.constant_initializer(0.1)
        layer_conv2d_1 = tf.layers.conv2d(
            inputs=inputs,
            filters=conv2d_filter,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=w_initializer,
            bias_initializer=bias_initializer,
            name="layer_conv2d_1"
        )
        layer_batch_normal_1 = tf.layers.batch_normalization(
            inputs=layer_conv2d_1,
            name="layer_batch_normal_1"
        )
        layer_conv2d_2 = tf.layers.conv2d(
            inputs=layer_batch_normal_1,
            filters=conv2d_filter,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=w_initializer,
            bias_initializer=bias_initializer,
            name="layer_conv2d_2"
        )
        layer_batch_normal_2 = tf.layers.batch_normalization(
            inputs=layer_conv2d_2,
            name="layer_batch_normal_2"
        )
        layer_max_pool_1 = tf.layers.max_pooling2d(
            inputs=layer_batch_normal_2,
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same",
            name="layer_max_pool_1"
        )
        layer_dropout_1 = tf.layers.dropout(
            inputs=layer_max_pool_1,
            rate=dropout_rate,
            name="layer_dropout_1"
        )
        return layer_dropout_1
