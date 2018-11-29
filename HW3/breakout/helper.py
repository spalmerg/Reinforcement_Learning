import numpy as np
import tensorflow as tf 
from skimage.color import rgb2gray
from skimage.transform import resize


def preprocess(img):
    img = np.uint8(resize(rgb2gray(img), (84,84))*255)
    return img

class Network():
    def __init__(self, learning_rate=0.01, hidden_size=10, action_size = 4, history_size=4, name="QEstimator"):
        with tf.variable_scope(name):
            # Set scope for copying purposes
            self.scope = name

            # Store Variables
            self.inputs_ = tf.placeholder(tf.float32, [None, 84, 84, history_size], name='inputs')
            self.target_preds_ = tf.placeholder(tf.float32, [None,], name="expected_future_rewards")
            self.chosen_action_pred = tf.placeholder(tf.float32, [None,], name="chosen_action_pred")
            self.actions_ = tf.placeholder(tf.int32, shape=[None], name='actions')
            self.avg_max_Q_ = tf.placeholder(tf.float32, name="avg_max_Q")
            self.reward_ = tf.placeholder(tf.float32, name="reward")

            # Normalizing the input
            self.inputscaled = self.inputs_/255.0
            
            # Three Convolutional Layers
            init = tf.variance_scaling_initializer(scale=2)
            self.conv1 = tf.contrib.layers.conv2d(self.inputscaled, 32, 8, 4, activation_fn=tf.nn.relu, 
                                                    padding='VALID',
                                                    weights_initializer=init)
            self.conv2 = tf.contrib.layers.conv2d(self.conv1, 64, 4, 2, activation_fn=tf.nn.relu,
                                                    padding='VALID',
                                                    weights_initializer=init)
            self.conv3 = tf.contrib.layers.conv2d(self.conv1, 64, 3, 1, activation_fn=tf.nn.relu,
                                                    padding='VALID',
                                                    weights_initializer=init)

            # Fully Connected Layers
            self.flatten = tf.contrib.layers.flatten(self.conv3)
            self.fc1 = tf.contrib.layers.fully_connected(self.flatten, hidden_size, 
                                                                 weights_initializer=init)
            self.predictions = tf.contrib.layers.fully_connected(self.fc1, action_size,
                                                                 weights_initializer=init, 
                                                                 activation_fn=None)
            
            # Get Prediction for the chosen action (epsilon greedy)
            self.action_one_hot = tf.one_hot(self.actions_, action_size, 1.0, 0.0, name='action_one_hot')
            self.chosen_action_pred = tf.reduce_sum(self.predictions * self.action_one_hot, reduction_indices=-1)

            # Calculate Loss
            delta = self.target_preds_ - self.chosen_action_pred
            self.loss = tf.reduce_mean(clipped_error(delta))
            
            # Adjust Network
            self.learn = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            # For Tensorboard
            with tf.name_scope("summaries"):
                tf.summary.scalar("loss", self.loss)
                tf.summary.scalar("avg_max_Q", self.avg_max_Q_)
                tf.summary.scalar("reward", self.reward_)
                self.summary_op = tf.summary.merge_all()
            
    def predict(self, sess, state):
        result = sess.run(self.predictions, feed_dict={self.inputs_: state})
        return result
    
    def update(self, sess, state, action, target_preds):
        feed_dict = {self.inputs_: state, 
                    self.actions_: action, 
                    self.target_preds_: target_preds}
        loss = sess.run([self.loss, self.learn], feed_dict=feed_dict)
        return loss

    def display(self, sess, state, action, target_preds, avg_max_Q, reward):
        feed_dict = {self.inputs_: state, 
                    self.actions_: action, 
                    self.target_preds_: target_preds,
                    self.avg_max_Q_: avg_max_Q, 
                    self.reward_: reward}
        loss = sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        return loss


def epsilon_greedy(sess, network, state, epsilon):
    pick = np.random.rand() # Uniform random number generator
    if pick < epsilon: # If off policy -- random action
        action = np.random.randint(0,4)
    else: # If on policy
        action = np.argmax(network.predict(sess, [state]))
    return action

def explore_prob(count, explore_start = 1, explore_stop = 0.1, decay_rate = 0.000001):
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * count)
    return(explore_probability)

def copy_parameters(sess, q_network, target_network):
    
    # Get and sort parameters
    q_params = [t for t in tf.trainable_variables() if t.name.startswith(q_network.scope)]
    q_params = sorted(q_params, key=lambda v: v.name)
    t_params = [t for t in tf.trainable_variables() if t.name.startswith(target_network.scope)]
    t_params = sorted(t_params, key=lambda v: v.name)
    
    # Assign Q-Parameters to Target Network
    updates = []
    for q_v, t_v in zip(q_params, t_params):
        update = t_v.assign(q_v)
        updates.append(update)
    
    sess.run(updates)

def clipped_error(x):
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)