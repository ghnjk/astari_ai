#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from memory import RandomMemory


def randomargmax(b, **kw):
    return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)


class DQN(object):

    def __init__(self, sess, feature_shape, action_count,
                 memory_size=10000,
                 batch_size=32,
                 update_network_iter=10000,
                 learning_rate=0.001,
                 reward_decay=0.99,
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
        self.tf_action = None
        # tf outputs
        self.tf_q_eval = None
        self.tf_q_real = None
        self.tf_loss = None
        # game estimate
        self.est_avg_reward = None
        self.est_game_count = 0
        self.est_game_sum_reward = 0.0
        # tf variables
        self.tf_eval_net_variables = None
        self.tf_target_net_variables = None
        # tf op
        self.tf_train_op = None
        self.tf_update_network_o = None
        self.tf_summaries = None

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
            return randomargmax(act_values)
        else:
            return np.random.randint(0, self.action_count)

    def store(self, cur_state, action, reward, next_state, is_done):
        item = [cur_state, action, reward, next_state, is_done]
        self.memory.append(item)
        self.data_count += 1

    def add_game_total_reward(self, r):
        self.est_game_sum_reward += r
        self.est_game_count += 1

    def learn(self, log_writer):
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
        is_done = []
        for item in samples:
            cur_state.append(item[0])
            action.append(item[1])
            reward.append(item[2])
            next_state.append(item[3])
            is_done.append(item[4])

        q_real = self.sess.run(
            self.tf_q_real,
            feed_dict={
                self.tf_next_state: next_state
            }
        )
        # 计算q_target = reward + reward_decay * max(q_real)
        q_target = np.zeros(shape=self.batch_size)
        for idx in range(self.batch_size):
            if is_done[idx]:
                q_target[idx] = reward[idx]
            else:
                q_target[idx] = reward[idx] + self.reward_decay * np.max(q_real[idx])

        _, loss = self.sess.run(
            [self.tf_train_op, self.tf_loss],
            feed_dict={
                self.tf_cur_state: cur_state,
                self.tf_q_target: q_target,
                self.tf_action: action
            }
        )
        if self.learn_counter % 5000 == 0:
            if self.est_game_count > 0:
                avg_r = self.est_game_sum_reward / self.est_game_count
            else:
                avg_r = 0.0
            # add sumary
            summaries = self.sess.run(self.tf_summaries, feed_dict={
                self.tf_cur_state: cur_state,
                self.tf_next_state: next_state,
                self.tf_q_target: q_target,
                self.tf_action: action,
                self.est_avg_reward: np.array([avg_r])
            })
            log_writer.add_summary(summaries, self.learn_counter)
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
            self.tf_q_target = tf.placeholder(tf.float32, [None, ], name="q_target")
            self.tf_action = tf.placeholder(tf.int32, [None, ], name="action")

        self.tf_q_eval = self._build_q_net(self.tf_cur_state, "eval_net")
        self.tf_q_real = self._build_q_net(self.tf_next_state, "target_net")

        with tf.variable_scope("evaluate_q"):
            batch_size = tf.shape(self.tf_cur_state)[0]
            a_indices = tf.stack([tf.range(batch_size, dtype=tf.int32), self.tf_action], axis=1)
            evaluate_q = tf.gather_nd(params=self.tf_q_eval, indices=a_indices)

        with tf.variable_scope("loss"):
            self.tf_loss = tf.reduce_mean(tf.squared_difference(self.tf_q_target, evaluate_q))

        with tf.variable_scope("train"):
            self.tf_train_op = tf.train.RMSPropOptimizer(
                self.learning_rate,
            ).minimize(self.tf_loss)

        self.tf_eval_net_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval_net")
        self.tf_target_net_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")
        with tf.variable_scope("update_network"):
            self.tf_update_network_o = [
                tf.assign(t, e) for t, e in zip(self.tf_target_net_variables, self.tf_eval_net_variables)
            ]

        self.est_avg_reward = tf.placeholder(tf.float32, [1], name="estimate_game_reward")

        self.tf_summaries = tf.summary.merge([
            tf.summary.scalar("game_avg_reward", self.est_avg_reward),
            tf.summary.scalar("loss", self.tf_loss),
            tf.summary.histogram("loss_hist", self.tf_loss),
            tf.summary.histogram("q_values_hist", self.tf_q_eval),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.tf_q_eval))
        ])

    def _build_q_net(self, inputs, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("image_feature"):
                with tf.variable_scope("conv2d_layers"):
                    layer_conv2d_1 = tf.layers.conv2d(
                        inputs=inputs,
                        filters=32,
                        kernel_size=(8, 8),
                        strides=(4, 4),
                        padding="valid",
                        activation=tf.nn.relu,
                        name="conv_layer_1"
                    )
                    layer_conv2d_2 = tf.layers.conv2d(
                        inputs=layer_conv2d_1,
                        filters=64,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        padding="valid",
                        activation=tf.nn.relu,
                        name="conv_layer_2"
                    )
                    layer_conv2d_3 = tf.layers.conv2d(
                        inputs=layer_conv2d_2,
                        filters=64,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding="valid",
                        activation=tf.nn.relu,
                        name="conv_layer_3"
                    )

                with tf.variable_scope("dense_layer_1"):
                    flattten_layer = tf.layers.flatten(
                        inputs=layer_conv2d_3,
                        name="flatter_layer"
                    )
                    dense_layer = tf.layers.dense(
                        inputs=flattten_layer,
                        units=512,
                        activation=None,
                        name="dense_layer"
                    )
                tf_img_feature = dense_layer
            q = tf.layers.dense(
                    inputs=tf_img_feature,
                    units=self.action_count,
                    activation=None,
                    name="prediction_layer"
                )
        return q

    def save_weight(self, weight_path):
        saver = tf.train.Saver()
        saver.save(self.sess, weight_path)

    def load_weights(self, weight_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, weight_path)