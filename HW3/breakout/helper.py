import numpy as np
import tensorflow as tf 

def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else just set to 1
    return np.reshape(image.astype(np.float).ravel(), [80,80])


class Network():
    def __init__(self, learning_rate=0.01, hidden_size=10, action_size = 2, memory_size=4, name="QEstimator"):
        with tf.variable_scope(name):
            # Set scope for copying purposes
            self.scope = name

            # Store Variables
            self.inputs_ = tf.placeholder(tf.float32, [None, 80, 80, memory_size], name='inputs')
            self.target_preds_ = tf.placeholder(tf.float32, [None,], name="expected_future_rewards")
            self.chosen_action_pred = tf.placeholder(tf.float32, [None,], name="chosen_action_pred")
            self.actions_ = tf.placeholder(tf.int32, shape=[None], name='actions')
            self.avg_reward_ = tf.placeholder(tf.float32, name="avg_reward")
            
            # Three Convolutional Layers
            init = tf.variance_scaling_initializer(scale=2)
            self.conv1 = tf.contrib.layers.conv2d(self.inputs_, 16, 8, 4, activation_fn=tf.nn.relu, 
                                                    padding='valid',
                                                    weights_initializer=init)
            self.conv2 = tf.contrib.layers.conv2d(self.conv1, 32, 4, 2, activation_fn=tf.nn.relu,
                                                    padding='valid',
                                                    weights_initializer=init)
            self.conv3 = tf.contrib.layers.conv2d(self.conv2, 64, 3, 1, activation_fn=tf.nn.relu,
                                                    padding='valid',  
                                                    weights_initializer=init)

            # Fully Connected Layers
            self.flatten = tf.contrib.layers.flatten(self.conv3)
            self.fc1 = tf.contrib.layers.fully_connected(self.flatten, hidden_size)
            self.predictions = tf.contrib.layers.fully_connected(self.fc1, action_size,
                                                                 weights_initializer=tf.contrib.layers.xavier_initializer(), 
                                                                 activation_fn=None)
            
            # Get Prediction for the chosen action (epsilon greedy)
            self.action_one_hot = tf.one_hot(self.actions_, action_size, 1.0, 0.0, name='action_one_hot')
            self.chosen_action_pred = tf.reduce_sum(self.predictions * self.action_one_hot, reduction_indices=-1)
    
            # Calculate Loss
            # self.losses = tf.squared_difference(self.target_preds_, self.chosen_action_pred)
            self.losses = tf.losses.huber_loss(self.target_preds_, self.chosen_action_pred)
            self.loss = tf.reduce_mean(self.losses)
            
            # Adjust Network
            self.learn = tf.train.RMSPropOptimizer(learning_rate).minimize(self.losses)

            # For Tensorboard
            with tf.name_scope("summaries"):
                tf.summary.scalar("loss", self.loss)
                tf.summary.scalar("avg_epoch_reward", self.avg_reward_)
                self.summary_op = tf.summary.merge_all()
            
    def predict(self, sess, state):
        result = sess.run(self.predictions, feed_dict={self.inputs_: state})
        return result
    
    def update(self, sess, state, action, target_preds, avg_reward):
        feed_dict = {self.inputs_: state, 
                    self.actions_: action, 
                    self.target_preds_: target_preds,
                    self.avg_reward_: avg_reward}
        loss = sess.run([self.loss, self.learn, self.summary_op], feed_dict=feed_dict)
        return loss


def epsilon_greedy(sess, network, state, epsilon=0.99):
    state = np.stack(list(state), axis=2)
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