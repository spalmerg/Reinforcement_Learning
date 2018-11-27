import gym
import gym.spaces
import tensorflow as tf
import numpy as np
import argparse
import logging
import os
import sys
from collections import deque
import random
from helper import preprocess, Network, epsilon_greedy, copy_parameters

os.environ["CUDA_VISIBLE_DEVICES"]="3"

parser = argparse.ArgumentParser(description='DQN for Breakout Game')

parser.add_argument('--learning_rate', default=0.00002, help='Learning rate for optimizer')
parser.add_argument('--discount_rate', default=0.95, help='Discount rate for future rewards')
parser.add_argument('--epochs', default=10000, help='Number of epochs to train')
parser.add_argument('--action_size', default=4, help='Number of actions in the game')
parser.add_argument('--hidden_size', default=512, help='Number of hidden neurons in FC layers')
parser.add_argument('--buffer_size', default=100000, help='Number of steps stored in the buffer')
parser.add_argument('--batch_size', default=800, help='Number of steps sampled from buffer')
parser.add_argument('--memory_size', default=4, help='Number of memory frames stored per state')
parser.add_argument('--reset_every', default=100, help='Number of steps before reset target network')
parser.add_argument('--update_every', default=4, help='Number of episodes to play before updating network')
parser.add_argument('--epsilon_explore', default=1500, help='Number of epochs to explore')
parser.add_argument('--epsilon_start', default=0.1, help='Start epsilon for epsilon greedy')
parser.add_argument('--epsilon_end', default=0.9, help='End epsilon for epsilon greedy')
parser.add_argument('--log_dir', default='logs/breakout/', help='Path to logs for tensorboard visualization')
parser.add_argument('--run_num', required=True, help='Provide a run number to correctly log')

def main(args):

    # create directory for logs
    if not os.path.exists(os.path.join(args.log_dir, args.run_num)):
        logging.info("Creating directory {0}".format(os.path.join(args.log_dir, args.run_num)))
        os.mkdir(os.path.join(args.log_dir, args.run_num))

    # set up and reset game environment
    env = gym.make('Breakout-v0')
    env.reset()

    # reset and initialize networks (Q & Target)
    tf.reset_default_graph()
    QNetwork = Network(name='QNetwork', hidden_size=args.hidden_size,
                                        learning_rate=args.learning_rate, 
                                        action_size=args.action_size,
                                        memory_size=args.memory_size)
    target = Network(name='Target', hidden_size=args.hidden_size,
                                        learning_rate=args.learning_rate, 
                                        action_size=args.action_size,
                                        memory_size=args.memory_size)

    # model saver
    saver = tf.train.Saver()

    # initialize buffer, state memory, and result
    buffer = deque(maxlen=args.buffer_size)
    state_memory = deque(maxlen=args.memory_size)

    # Train the DQN
    with tf.Session() as sess: 
        writer = tf.summary.FileWriter(os.path.join(args.log_dir, args.run_num), sess.graph)
        # restore checkpoint
        saver.restore(sess, "model/model900.ckpt")
        # start new model
        #sess.run(tf.global_variables_initializer())
        
        # Set up count for network reset
        count = 0
        
        # Make epsilon greedy schedule
        epsilons = np.linspace(args.epsilon_start, args.epsilon_end, args.epsilon_explore)
        epsilons = list(epsilons) + list(np.repeat(args.epsilon_end, args.epochs - len(epsilons)))
        
        # Set up memory for episode
        start_state = preprocess(env.reset())
        for fill in range(args.memory_size):
            state_memory.append(start_state)

        # Fill The Buffer
        for i in range(args.buffer_size//2):
            # take action based on state & epsilon greedy
            action = epsilon_greedy(sess, QNetwork, state_memory)
            new_state, reward, done, _ = env.step(action)

            # Save old_state_memory
            old_state_memory = state_memory

            # Add new_state to state_memory
            state_memory.append(preprocess(new_state))

            # Add step to buffer
            old_state_memory_reshape = np.reshape(old_state_memory, [80, 80, args.memory_size])
            state_memory_reshape = np.reshape(state_memory, [80, 80, args.memory_size])
            buffer.append([old_state_memory_reshape, action, state_memory_reshape, reward, done])

            # If done, reset memory
            if done:
                start_state = preprocess(env.reset())
                for fill in range(args.memory_size):
                    state_memory.append(start_state)
        
        # Initialize result for reporting
        result = []

        # Train
        for epoch in range(900, args.epochs):
            result = []
            
            # Set Up Memory
            start_state = preprocess(env.reset())
            for fill in range(args.memory_size):
                state_memory.append(start_state)

            while True: 
                # Add M to buffer (following policy)
                action = epsilon_greedy(sess, QNetwork, state_memory, epsilons[epoch])
                new_state, reward, done, _ = env.step(action)

                # Save old_state_memory
                old_state_memory = state_memory

                # Add New_state to state_memory
                state_memory.append(preprocess(new_state))
            
                # Add step to buffer
                old_state_memory_reshape = np.reshape(old_state_memory, [80, 80, args.memory_size])
                state_memory_reshape = np.reshape(state_memory, [80, 80, args.memory_size])
                buffer.append([old_state_memory_reshape, action, state_memory_reshape, reward, done])
                
                # If simulation done, stop
                if done:
                    break
                
                ### Sample & Update
                if count % args.update_every == 0:
                    sample = random.sample(buffer, args.batch_size)
                    state_b, action_b, new_state_b, reward_b, done_b = map(np.array, zip(*sample))

                    # Find max Q-Value per batch for progress
                    Q_preds = sess.run(QNetwork.chosen_action_pred, 
                                        feed_dict={QNetwork.inputs_: state_b,
                                        QNetwork.actions_: action_b})
                    result.append(np.max(Q_preds))

                    # Q-Network
                    T_preds = []
                    TPreds_batch = target.predict(sess, new_state_b)
                    for i in range(args.batch_size):
                        terminal = done_b[i]
                        if terminal:
                            T_preds.append(reward_b[i])
                        else:
                            T_preds.append(reward_b[i] + args.discount_rate * np.max(TPreds_batch[i]))

                    # Update Q-Network
                    avg_max_Q = np.mean(result)
                    loss, _, summary = QNetwork.update(sess, state_b, action_b, T_preds, avg_max_Q)
                
                # Save target network parameters every epoch
                count += 1
                if count % args.reset_every == 0:
                    copy_parameters(sess, QNetwork, target)

            # Log and save models
            logger.info("Epoch: {0}\tAvg Reward: {1}".format(epoch, np.mean(result)))
            writer.add_summary(summary, epoch)
            if epoch % 20 == 0:
                    saver.save(sess, "./model/model{0}.ckpt".format(epoch))

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    args = parser.parse_args()
    main(args)