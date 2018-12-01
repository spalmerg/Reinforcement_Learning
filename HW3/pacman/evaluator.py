import gym
import gym.spaces
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from helper import preprocess, Network
from collections import deque
import os

# GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Arguments
parser = argparse.ArgumentParser(description='DQN for Breakout Game')
parser.add_argument('--hidden_size', default=512, help='Number of hidden neurons in FC layers')
parser.add_argument('--learning_rate', default=0.001, help='Learning rate for optimizer')
parser.add_argument('--action_size', default=4, help='Number of actions in the game')
parser.add_argument('--games', default=1000, help="Number of games to play")
parser.add_argument('--history_size', default=4, help='Number of steps sampled from buffer')


def main(args): 
    # set up and reset game environment 
    env = gym.make('Breakout-v0')
    env.reset()

    # initialize networks
    QNetwork = Network(name='QNetwork', hidden_size=args.hidden_size,
                                        learning_rate=args.learning_rate, 
                                        action_size=args.action_size)
    target = Network(name='Target', hidden_size=args.hidden_size,
                                        learning_rate=args.learning_rate, 
                                        action_size=args.action_size)

    saver = tf.train.Saver()

    # initialize history
    history = np.zeros((80, 80, args.history_size + 1), dtype=np.uint8)

    # load game
    with tf.Session() as sess:
        print("SESSION STARTED")
        saver.restore(sess, "model/model15500.ckpt")
        print("MODEL RESTORED")
        for game in range(args.games):
            # start game & initialize memory
            state = env.reset()
            for i in range(5):
                history[:, :, i] = preprocess(state)

            reward_total = 0
            while True: 
                action = np.argmax(QNetwork.predict(sess, [history[:,:,:args.history_size]]))
                new_state, reward, done, _ = env.step(action)

                # history updates
                history[:,:, args.history_size] = preprocess(new_state) # add new state at end
                history[:,:,:args.history_size] = history[:,:,1:] # shift history

                if done:
                    break
                else: 
                    # Update reward total
                    reward_total = reward_total + reward

            print(reward_total)  

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)