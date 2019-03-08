"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.
The cart pole example. Policy is oscillated.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

# np.random.seed(2)
# tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 300
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 50   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9    # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic


N_F = state_size = 3
N_A = action_size = 3


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units= 180,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=180,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error



def env_step(s,a): # a = -100, -75, -50, -25, 0, 25, 50, 75, 100
    # prices = [278,272,274,272,268,276,292,302,312,318,322,326,328,326,324,316,300,290,290,300,310,316,302,298]
    prices = np.arange(24)
    done = False #done
    # dis_max = char_max=0
    interval = s[0] # time interval step hour
    soc = s[1] #engergy step %
    # 10% uncertainty of prices
    price = s[2]#*(1+np.random.uniform(-0.05,0.05)) # price step $1
    s1 = np.array([0,0,0]) # next state

    if a==0: # discharge
        if(soc==0):
            s1=s
            # r=-500
            r = -1000
            # r = 0
        else:
            s1[0]=interval+1
            # dis_max = soc
            r = price * 10 # reward
            s1[1] = soc - 10 # energy reduce 10%
            s1[2] = prices[s1[0]] # update new price by time interval
    elif a==1: # keep same, do nothing
        s1[0]=interval+1
        s1[1]=soc
        s1[2]=prices[s1[0]]
        r=0
    else: # a==2 ,charge
        if(soc==100):
            s1=s # keep old state
            # r=-500 # penalty
            r = -1000
        else:
            s1[0]=interval+1
            # char_max = 100-soc
            r = -price * 10 # penalty
            s1[1]= soc + 10  # energy reduce 10%
            s1[2]=prices[s1[0]]
    if (s1[0] >= 23):
        done = True        
    return s1,r,done



sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())


reward_all_episodes =[]
for i_episode in range(MAX_EPISODE):
    
    # s = np.array([0,50,278])
    # time = np.array.
    s = np.array([0, 60, 250])

    t = 0
    track_r = []
    while True:
       

        a = actor.choose_action(s)
        print(a)

        s_, r, done = env_step(s, a)

        track_r.append(r)

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        print('***************** Error: ', td_error)
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            reward_all_episodes.append(running_reward)
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            #print("episode:", i_episode, "  reward:", int(running_reward))
            break

plt.plot(reward_all_episodes)
plt.ylabel('rewards')
plt.show()