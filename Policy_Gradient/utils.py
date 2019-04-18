import torch
import os
import gym
from gym_recording.wrappers import TraceRecordingWrapper
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def save_weights(policy, args, directory='./preTrained'):
    # see if the folder exit if note create one
    create_folder(directory)
    print("Saving weights")
    torch.save(policy.state_dict(), '{}/{}_{}.pth'.format(directory, args.algo, args.env_name))



def load_weights(policy, args, directory='./preTrained'):
    if os.path.exists(directory):
        print("Loading PreTrained Weights")
        policy.load_state_dict(torch.load('{}/{}_{}.pth'.format(directory, args.algo, args.env_name)))
    else:
        print("PreTrained Weights don't exists. Training Agent from scratch")



def save_results(store, args, keys, session):
    df = pd.DataFrame.from_records(store)
    columns = [key for key in keys]
    df.columns = columns

    # create directory
    #dir = './Results/{}/{}_{}'.format(args.algo, args.env_name, time.strftime('%y-%m-%d-%H-%M-%s'))
    dir = './Results/{}/{}_{}'.format(args.algo, args.env_name, session)
    create_folder(dir)   # create_folder (directory)

    # save csv
    df.to_csv('{}.csv'.format(dir))
    # save graph
    plot_graph(df, dir)
    print('saved_eval_results')

def plot_graph(results, dir):
    plt.plot(results.Iteration, results.Total_Reward, label='Reward over trajectory')
    plt.xlabel("No of Iterations")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig('{}.svg'.format(dir), format='svg', dpi=1200)
    plt.close()

def test(policy, args):
    load_weights(policy, args)
    env = gym.make(args.env_name)
    if not os.path.exists(args.env_name):
        env = gym.wrappers.Monitor(env, directory='./Results/{}/{}'.format(args.algo, args.env_name), force=True, write_upon_reset=True)

    for i_episode in range(10):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:  # Don't infinite loop while learning
            env.render()
            action, action_log_prob = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        print('Iteration = {} \t Total Reward = {}'.format(i_episode,total_reward))
    env.close()

def eval(env, policy, args):
    score = 0
    for i_episode in range(20):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:  # Don't infinite loop while learning
            #env.render()
            action, action_log_prob = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        score += total_reward
    score /= i_episode
    #if score > policy.best_policy_reward:
    #    policy.best_policy_reward = score
    #    save_weights(policy, args)
    #    print("Saving the best found policy")
    return score