import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np
from simulation_env import Env
import scipy.io as sio
import tqdm
import pickle as pk
import sys
import os

LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 500
REPLACE_ITER_C = 300
MEMORY_CAPACITY = 20000  # 7000
BATCH_SIZE = 32

tf.disable_v2_behavior()


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, ):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q  论文中是朝着Q最大化的方向更新参数 这里达到的是同样的效果
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        if s.shape[0] >= 1:
            return self.sess.run(self.a, {self.S: s})
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self, replace):
        # hard replace parameters
        if self.a_replace_counter % REPLACE_ITER_A == 0 or replace:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)])
            # print('Actor target network replaced with eval net')
        if self.c_replace_counter % REPLACE_ITER_C == 0 or replace:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)])
            # print('Critic target network replaced with eval network')
        self.a_replace_counter += 1
        self.c_replace_counter += 1

        if self.pointer < MEMORY_CAPACITY:
            indices = np.random.choice(self.pointer, size=BATCH_SIZE)
        else:
            indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})  # {}place holder 处理输入

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    def savenet(self, file):
        saver = tf.train.Saver()
        saver.save(self.sess, 'save/' + file + '.ckpt')

    def restore(self, file):
        saver = tf.train.Saver()
        saver.restore(self.sess, file + '.ckpt')

    def print_weight(self):
        tvars = tf.trainable_variables()
        tvars_vals = self.sess.run(tvars)

        for var, val in zip(tvars, tvars_vals):
            if var.name == 'Actor/eval/l1/kernel:0':
                l1_w = val
            elif var.name == 'Actor/eval/l1/bias:0':
                l1_b = val
            elif var.name == 'Actor/eval/a/kernel:0':
                a_w = val
            elif var.name == 'Actor/eval/a/bias:0':
                a_b = val
        return l1_w, l1_b, a_w, a_b
