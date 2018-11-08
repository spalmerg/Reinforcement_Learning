from PongClasses import PolicyGradient, Baseline
from PongFunctions import preprocess, expected_rewards
import gym
import gym.spaces
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

#### set up GPU
os.environ["CUDA_VISIBLE_DEVICES"]="2"

##### set parameters
episodes = 4
gamma = 0.99 # specified by homework
learning_rate = 0.002
hidden_size = 20 
epochs = 2


if __name__ == "__main__":
    # reset tensorflow graph
    tf.reset_default_graph()

    # initialize networks
    network = PolicyGradient(name = 'pray4us', hidden_size=hidden_size, learning_rate=learning_rate)
    baseline = Baseline(name = 'ughvariance', hidden_size=hidden_size, learning_rate=learning_rate)

    # set up pong environment
    env = gym.make('Pong-v0')

    # Initialize the simulation
    env.reset()

    # store epoch rewards
    all_rewards = []

    # train Policy Gradient Network
    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
            
            # save all states, actions, and rewards that happen 
            episode_states, episode_actions, all_discount_rewards = [], [], []
            total_epoch_reward = 0
            running_rewards = []
            
            for ep in range(episodes):
                
                state = preprocess(env.reset())
                episode_rewards = []
                
                while True: 
                
                    # get action prob distribution w.r.t. policy
                    feed = {network.inputs_: state.reshape((1,*state.shape))}
                    action_prob_dist = sess.run(network.action_distribution, feed_dict=feed)
                    
                    # select action w.r.t. distribution, only RIGHT & LEFT
                    action = np.random.choice([2,3], p=action_prob_dist.ravel())
                    new_state, reward, done, _ = env.step(action)
                    
                    # keep track of all states, actions, and rewards
                    episode_states.append(state)
                    episode_rewards.append(reward)
                    
                    # reformat action for softmax
                    action_ = np.zeros(action_prob_dist.shape[1]) # 6 Pong has 6 action space
                    action_[action - 2] = 1
                    episode_actions.append(action_)
                    
                    # reset current state to be new state
                    state = preprocess(new_state)
                    
                    if done:
                        # Calculate discounted reward per episode
                        exp_rewards = expected_rewards(episode_rewards)
                        all_discount_rewards += exp_rewards
                        
                        # reward per episode
                        running_rewards.append(sum(episode_rewards))
                        break
            
            # get baseline adjustment
            baseline_ = sess.run(baseline.fc3, feed_dict={baseline.inputs_ : np.vstack(episode_states)})
            exp_rewards_b = all_discount_rewards - np.hstack(baseline_)
            
            # train baseline network
            _, _= sess.run([baseline.loss, baseline.learn], 
                        feed_dict={baseline.inputs_: np.vstack(episode_states),
                        baseline.expected_future_rewards_: all_discount_rewards })
            
            # update Policy Gradient Network
            # if interested in seeing without baseline correction exp_rewards_b --> all_discount_rewards
            _, _= sess.run([network.loss, network.learn], 
                                        feed_dict={network.inputs_: np.vstack(episode_states),
                                        network.actions_: np.vstack(episode_actions),
                                        network.expected_future_rewards_: exp_rewards_b })      
            
            # average reward per episodes in epoch
            all_rewards.append(np.mean(running_rewards))
            print("Epoch: %s    Average Reward: %s \n" %(epoch, np.mean(running_rewards)))   
