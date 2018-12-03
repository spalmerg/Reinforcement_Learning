import numpy as np
import tensorflow as tf 


def preprocess(img):
    img = img[::2, ::2]
    img = np.mean(img, axis=2).astype(np.uint8)
    return img

class Network():
    def __init__(self, 
                 learning_rate=0.00025,
                 learning_rate_start=.001,
                 learning_rate_decay = 0.96, 
                 learning_rate_decay_step = 1000000, 
                 hidden_size=10, 
                 action_size = 4, 
                 history_size=4, 
                 name="Network"):

        with tf.variable_scope(name):
            # Set scope for copying purposes
            self.scope = name

            # Initializers
            conv_init = tf.contrib.layers.xavier_initializer_conv2d()
            init = tf.contrib.layers.xavier_initializer()

            # Store Variables
            self.inputs_ = tf.placeholder(tf.float32, [None, 105, 80, history_size], name='inputs')
            self.target_preds_ = tf.placeholder(tf.float32, [None,], name="expected_future_rewards")
            self.chosen_action_pred = tf.placeholder(tf.float32, [None,], name="chosen_action_pred")
            self.actions_ = tf.placeholder(tf.int32, shape=[None], name='actions')
            self.avg_max_Q_ = tf.placeholder(tf.float32, name="avg_max_Q")
            self.reward_ = tf.placeholder(tf.float32, name="reward")
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')

            # Normalizing the input
            self.inputs_scaled_ = self.inputs_/255.0
            
            # Three Convolutional Layers
            self.conv1 = tf.layers.conv2d(
                inputs = self.inputs_scaled_, 
                filters = 16,
                kernel_size = [8,8],
                strides = [4,4],
                padding = "VALID",
                kernel_initializer=conv_init,
                activation=tf.nn.relu)
            self.conv2 = tf.layers.conv2d(
                inputs = self.conv1, 
                filters = 8,
                kernel_size = [4,4],
                strides = [2,2],
                padding = "VALID",
                kernel_initializer=conv_init,
                activation=tf.nn.relu)
            # self.conv3 = tf.layers.conv2d(
            #     inputs = self.conv2, 
            #     filters = 128,
            #     kernel_size = [4,4],
            #     strides = [2,2],
            #     padding = "VALID",
            #     kernel_initializer=conv_init,
            #     activation=tf.nn.relu)

            # Fully Connected Layers
            self.flatten = tf.contrib.layers.flatten(self.conv2)
            self.fc1 = tf.layers.dense(self.flatten, 
                                        hidden_size, 
                                        activation=tf.nn.relu,
                                        kernel_initializer=init)
            self.predictions = tf.layers.dense(self.fc1, 
                                        action_size, 
                                        activation=None,
                                        kernel_initializer=init)
            
            # Get Prediction for the chosen action (epsilon greedy)
            self.action_one_hot = tf.one_hot(self.actions_, action_size, 1.0, 0.0, name='action_one_hot')
            self.chosen_action_pred = tf.reduce_sum(tf.multiply(self.predictions, self.action_one_hot), axis=1)

            # Calculate Loss
            self.losses = tf.losses.huber_loss(self.target_preds_, self.chosen_action_pred)
            self.loss = tf.reduce_mean(self.losses)
            
            # Adjust Network
            self.learning_rate_op = tf.maximum(learning_rate, tf.train.exponential_decay(
                                                learning_rate_start,
                                                self.learning_rate_step,
                                                learning_rate_decay_step,
                                                learning_rate_decay,
                                                staircase=True))
            self.learn = tf.train.AdamOptimizer(self.learning_rate_op).minimize(self.loss)

            # For Tensorboard
            with tf.name_scope("summaries"):
                tf.summary.scalar("loss", self.loss)
                tf.summary.scalar("avg_max_Q", self.avg_max_Q_)
                tf.summary.scalar("reward", self.reward_)
                self.summary_op = tf.summary.merge_all()
            
    def predict(self, sess, state):
        result = sess.run(self.predictions, feed_dict={self.inputs_: state})
        return result
    
    def update(self, sess, state, action, target_preds, global_step):
        feed_dict = {self.inputs_: state, 
                    self.actions_: action, 
                    self.target_preds_: target_preds, 
                    self.learning_rate_step: global_step}
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
    if pick > epsilon: # If off policy -- random action
        action = np.random.randint(0,4)
    else: # If on policy
        action = np.argmax(network.predict(sess, [state]))
    return action

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