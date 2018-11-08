# policy gradient class
class PolicyGradient():
    def __init__(self, learning_rate=0.01, state_size=6400, action_size=2, hidden_size=20, name='PolicyGradient'):
        with tf.variable_scope(name):
            
            # Store Variables
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')
            self.actions_ = tf.placeholder(tf.int32, [None, action_size], name='actions')
            self.expected_future_rewards_ = tf.placeholder(tf.float32, [None,], name="expected_future_rewards")
            
            # Hidden Layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size, 
                                                         weights_initializer=tf.contrib.layers.xavier_initializer())
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, action_size, 
                                                         weights_initializer=tf.contrib.layers.xavier_initializer())
            self.fc3 = tf.contrib.layers.fully_connected(self.fc2, action_size, activation_fn=None, 
                                                         weights_initializer=tf.contrib.layers.xavier_initializer())
            
            # Output Layer
            self.action_distribution = tf.nn.softmax(self.fc3)
            
            # Training Section
            self.log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.fc3, labels = self.actions_)
            self.loss = tf.reduce_mean(self.log_prob * self.expected_future_rewards_)

            # Adjust Network
            self.learn = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


class Baseline():
    def __init__(self, learning_rate=0.01, state_size=6400, hidden_size=10, name="Baseline"):
        with tf.variable_scope(name):

            # Store Variables
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')
            self.expected_future_rewards_ = tf.placeholder(tf.float32, [None,], name="expected_future_rewards")

            # Hidden Layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size, 
                                                         weights_initializer=tf.contrib.layers.xavier_initializer())
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size, 
                                                         weights_initializer=tf.contrib.layers.xavier_initializer())
            self.fc3 = tf.contrib.layers.fully_connected(self.fc2, 1, activation_fn=None, 
                                                                 weights_initializer=tf.contrib.layers.xavier_initializer())

            # Define Loss
            self.loss = tf.reduce_mean(tf.square(self.fc3 - self.expected_future_rewards_), name="mse")

            # Adjust Network
            self.learn = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)