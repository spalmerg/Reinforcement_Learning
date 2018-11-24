import gym
import gym.spaces
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import helper

def main(): 
    # initialize networks
    tf.reset_default_graph()
    QNetwork = Network(name = 'QNetwork', hidden_size=hidden_size, learning_rate=learning_rate)
    target = Network(name = 'Target', hidden_size=hidden_size, learning_rate=learning_rate)

    # saver
    saver = tf.train.Saver()
    all_rewards = []

    # load game
    with tf.Session() as sess:
        saver.restore(sess, 'model/model1000.ckpt.meta') ## add last checkpoint
        for game in range(1000):
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