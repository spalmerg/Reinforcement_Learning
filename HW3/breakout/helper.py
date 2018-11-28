import numpy as np
import tensorflow as tf 


def preprocess(img):
    img = img[::2, ::2]
    img = np.mean(img, axis=2).astype(np.uint8)
    return img

class Network():
    def __init__(self, learning_rate=0.01, hidden_size=10, action_size = 4, history_size=4, name="QEstimator"):
        with tf.variable_scope(name):
            # Set scope for copying purposes
            self.scope = name

            # Store Variables
            self.inputs_ = tf.placeholder(tf.float32, [None, 105, 80, history_size], name='inputs')
            self.target_preds_ = tf.placeholder(tf.float32, [None,], name="expected_future_rewards")
            self.chosen_action_pred = tf.placeholder(tf.float32, [None,], name="chosen_action_pred")
            self.actions_ = tf.placeholder(tf.int32, shape=[None], name='actions')
            self.avg_max_Q_ = tf.placeholder(tf.float32, name="avg_max_Q")
            self.reward_ = tf.placeholder(tf.float32, name="reward")

            # Normalizing the input
            self.inputscaled = self.inputs_/255.0
            
            # Three Convolutional Layers
            init = tf.variance_scaling_initializer(scale=2)
            self.conv1 = tf.contrib.layers.conv2d(self.inputs_, 16, 8, 4, activation_fn=tf.nn.relu, 
                                                    padding='VALID',
                                                    weights_initializer=init)
            self.conv2 = tf.contrib.layers.conv2d(self.conv1, 32, 4, 2, activation_fn=tf.nn.relu,
                                                    padding='VALID',
                                                    weights_initializer=init)

            # Fully Connected Layers
            self.flatten = tf.contrib.layers.flatten(self.conv2)
            self.fc1 = tf.contrib.layers.fully_connected(self.flatten, hidden_size, 
                                                                 weights_initializer=init)
            self.predictions = tf.contrib.layers.fully_connected(self.fc1, action_size,
                                                                 weights_initializer=init, 
                                                                 activation_fn=None)
            
            # Get Prediction for the chosen action (epsilon greedy)
            self.indices = tf.range(1) * tf.size(self.actions_) + self.actions_
            self.chosen_action_pred = tf.gather(tf.reshape(self.predictions, [-1]), self.indices)
    
            # Calculate Loss
            self.losses = tf.losses.huber_loss(self.target_preds_, self.chosen_action_pred)
            self.loss = tf.reduce_mean(self.losses)
            
            # Adjust Network
            self.learn = tf.train.AdamOptimizer(learning_rate).minimize(self.losses)

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


def epsilon_greedy(sess, network, state, epsilon=0.9):
    pick = np.random.rand() # Uniform random number generator
    if pick > epsilon: # If off policy -- random action
        action = np.random.randint(0,4)
    else: # If on policy
        action = np.argmax(network.predict(sess, [state]))
    return action


def copy_parameters(sess, q_network, target_network):

    q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_network.scope)
    t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_network.scope)

    sess.run([v_t.assign(v) for v_t, v in zip(t_vars, q_vars)])