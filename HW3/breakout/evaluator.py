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
parser.add_argument('--learning_rate', default=0.000025, help='Learning rate for optimizer')
parser.add_argument('--action_size', default=4, help='Number of actions in the game')
parser.add_argument('--games', default=1000, help="Number of games to play")


def main(args): 
    # set up and reset game environment 
    env = gym.make('Breakout-v0')
    state_memory = deque(maxlen=args.memory_size)
    env.reset()

    # initialize networks
    QNetwork = Network(name='QNetwork', hidden_size=args.hidden_size,
                                        learning_rate=args.learning_rate, 
                                        action_size=args.action_size)
    target = Network(name='Target', hidden_size=args.hidden_size,
                                        learning_rate=args.learning_rate, 
                                        action_size=args.action_size)

    saver = tf.train.Saver()

    # load game
    with tf.Session() as sess:
        print("SESSION STARTED")
        saver.restore(sess, "model/model860.ckpt")
        print("MODEL RESTORED")
        for game in range(args.games):
            # start game & initialize memory
            state = preprocess(env.reset())

            reward_total = 0
            while True: 
                action = np.argmax(QNetwork.predict(sess, [state]))
                new_state, reward, done, _ = env.step(action)

                if done:
                    break
                else: 
                    # Update reward total
                    reward_total = reward_total + reward
                    # Update state
                    state = preprocess(new_state)


            print(reward_total)  

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)