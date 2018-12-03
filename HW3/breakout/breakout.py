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

os.environ["CUDA_VISIBLE_DEVICES"]="2"

parser = argparse.ArgumentParser(description='DQN for Breakout Game')

parser.add_argument('--learning_rate', default=0.0001, help='Learning rate for optimizer')
parser.add_argument('--discount_rate', default=0.95, help='Discount rate for future rewards')
parser.add_argument('--epochs', default=10000000000, help='Number of epochs to train')
parser.add_argument('--action_size', default=4, help='Number of actions in the game')
parser.add_argument('--hidden_size', default=512, help='Number of hidden neurons in FC layers')
parser.add_argument('--buffer_size', default=1000000, help='Number of steps stored in the buffer')
parser.add_argument('--batch_size', default=32, help='Number of steps sampled from buffer')
parser.add_argument('--history_size', default=4, help='Number of steps sampled from buffer')
parser.add_argument('--reset_every', default=10000, help='Number of steps before reset target network')
parser.add_argument('--update_every', default=4, help='Number of steps before reset target network')
parser.add_argument('--log_dir', default='logs/breakout/', help='Path to logs for tensorboard visualization')
parser.add_argument('--run_num', required=True, help='Provide a run number to correctly log')
parser.add_argument('--epsilon_explore', default=1000000, help='Number of frames to explore')
parser.add_argument('--epsilon_start', default=0.1, help='Start epsilon for epsilon greedy')
parser.add_argument('--epsilon_end', default=0.9, help='End epsilon for epsilon greedy')


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
                                        history_size=args.history_size)
    target = Network(name='Target', hidden_size=args.hidden_size,
                                        learning_rate=args.learning_rate, 
                                        action_size=args.action_size,
                                        history_size=args.history_size)

    # model saver
    saver = tf.train.Saver()

    # initialize buffer & history
    buffer = deque(maxlen=args.buffer_size)
    history = np.zeros((105, 80, args.history_size + 1), dtype=np.uint8)

    # exploit/explore schedule
    epsilons = np.linspace(args.epsilon_start, args.epsilon_end, args.epsilon_explore)
    epsilons = list(epsilons) + list(np.repeat(args.epsilon_end, 1e7))

    # Train the DQN
    with tf.Session() as sess: 
        writer = tf.summary.FileWriter(os.path.join(args.log_dir, args.run_num), sess.graph)
        # restore checkpoint
        # saver.restore(sess, "model1/model880.ckpt")
        # start new model
        sess.run(tf.global_variables_initializer())
        
        # Set up count for network reset
        count = 0

        # Set up history for episode
        state = env.reset()
        for i in range(5):
            history[:, :, i] = preprocess(state)

        # Fill The Buffer
        for i in range(args.buffer_size//20):
            action = epsilon_greedy(sess, QNetwork, history[:,:,:args.history_size], epsilons[count])
            new_state, reward, done, _ = env.step(action)
            
            # history updates
            history[:,:, args.history_size] = preprocess(new_state) # add new state at end
            history[:,:,:args.history_size] = history[:,:,1:] # shift history
            new_state = np.copy(history[:,:,1:]) # with new state
            old_state = np.copy(history[:,:,:args.history_size]) # without new state
            
            # Add step to buffer
            buffer.append([old_state, action, new_state, reward, done])
            
            # If done, reset history
            if done: 
                state = env.reset()
                for i in range(args.history_size + 1):
                    history[:, :, i] = preprocess(state)
                break

        # Train
        for epoch in range(args.epochs):
            # For Tensorboard
            result = [] 
            reward_total = 0
            
            # Set Up Memory
            state = env.reset() # get first state
            for i in range(args.history_size + 1):
                history[:, :, i] = preprocess(state)

            while True: 
                # Add M to buffer (following policy)
                action = epsilon_greedy(sess, QNetwork, history[:,:,:args.history_size], epsilons[count])
                new_state, reward, done, _ = env.step(action)

                # deal with history and state
                history[:,:, args.history_size] = preprocess(new_state) # add new state at end
                history[:,:,:args.history_size] = history[:,:,1:] # shift history
                new_state = np.copy(history[:,:,1:]) # with new state
                old_state = np.copy(history[:,:,:args.history_size]) # without new state

                # Add step to buffer
                buffer.append([old_state, action, new_state, reward, done])

                if count % args.update_every == 0: 
                    ### Sample & Update
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
                    loss, _ = QNetwork.update(sess, state_b, action_b, T_preds, count)

                # If simulation done, stop
                if done:
                    # Reset history
                    state = env.reset()
                    for i in range(args.history_size + 1):
                        history[:, :, i] = preprocess(state)
                    # Tensorboard
                    avg_max_Q = np.mean(result)
                    loss, summary = QNetwork.display(sess, state_b, action_b, T_preds, avg_max_Q, reward_total)
                    # Log and save models
                    logger.info("Epoch: {0}\tAvg Reward: {1}".format(epoch, avg_max_Q))
                    writer.add_summary(summary, epoch)
                    break
                else: 
                    reward_total = reward_total + reward
                
                # Save target network parameters every epoch
                count += 1
                if count % args.reset_every == 0:
                    copy_parameters(sess, QNetwork, target)

            # save model
            if epoch % 20 == 0:
                    saver.save(sess, "./model/model{0}.ckpt".format(epoch))

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    args = parser.parse_args()
    main(args)