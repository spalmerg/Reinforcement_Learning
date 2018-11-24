import gym
import gym.spaces
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from helper import preprocess, Network

parser = argparse.ArgumentParser(description='DQN for Breakout Game')

parser.add_argument('--hidden_size', default=200, help='Number of hidden neurons in FC layers')
parser.add_argument('--learning_rate', default=0.0005, help='Learning rate for optimizer')
parser.add_argument('--action_size', default=4, help='Number of actions in the game')
parser.add_argument('--games', default=1000, help="Number of games to play")


def main(args): 
    # initialize networks
    tf.reset_default_graph()
    QNetwork = Network(name='QNetwork', hidden_size=args.hidden_size,
                                        learning_rate=args.learning_rate, 
                                        action_size=args.action_size)
    target = Network(name='Target', hidden_size=args.hidden_size,
                                        learning_rate=args.learning_rate, 
                                        action_size=args.action_size)

    # saver
    saver = tf.train.Saver()
    all_rewards = []

    # load game
    with tf.Session() as sess:
        saver.restore(sess, 'model/model1000.ckpt.meta') ## add last checkpoint
        for game in range(args.games):
            # start game
            state = preprocess(env.reset())
            reward_total = 0
            while True: 
                action = np.argmax(QNetwork.predict(sess, [state]))
                new_state, reward, done, _ = env.step(action)
                
                if done:
                    break
                else: 
                    state = preprocess(new_state)
                    reward_total += reward
                    
            all_rewards.append(reward_total)
        fig = plt.hist(all_rewards)
        plt.title('Breakout Rewards (1000 Games)')
        plt.xlabel('Number of Games')
        plt.ylabel('Reward Scored')
        plt.savefig('Breakout_Rewards.png')

if __name__ == "__main__":
    main()