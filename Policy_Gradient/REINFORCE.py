#link: https://www.datahubbs.com/reinforce-with-pytorch/
#link: https://www.datahubbs.com/policy-gradients-with-reinforce/


import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import utils
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--algo', default='REINFORCE')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--normalize_reward', action='store_true',
                    help='normalize discounted reward')
parser.add_argument('--env_name', default='CartPole-v0')
parser.add_argument('--max_itr', type=int, default=2000)
parser.add_argument('--Test_run', default=True, type=bool)





class Policy(nn.Module):
    def __init__(self, env):
        super(Policy, self).__init__()
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, self.output_dim)

        self.saved_log_probs = []
        self.rewards = []
        self.best_policy_reward = 0

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        action_scores = self.fc2(x)
        return torch.softmax(action_scores, dim=1)

    def update_weights(self, episodic_loss):
        self.optimizer.zero_grad()
        episodic_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        """sample A_t ~ policy"""
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()


def compute_episodic_loss(policy, args):
    R = 0
    policy_loss = []
    rewards = []

    """calculates discounted reward"""
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)

    """Normalizes the reward"""
    if args.normalize_reward == True:
        eps = np.finfo(np.float32).eps.item()
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    """Calculates loss"""
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)

    # Comment: We could try for mean as well
    policy_loss = torch.cat(policy_loss).sum()
    return policy_loss



def REINFORCE():
    args = parser.parse_args()
    env = gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = Policy(env)

    #running_reward = 10
    store = []
    for i_episode in range(args.max_itr):
        state = env.reset()
        total_reward = 0
        for t in range(10000):  # Don't infinite loop while learning
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            total_reward += reward
            if done:
                break

        #running_reward = running_reward * 0.99 + t * 0.01

        # Compute episodic loss and update weights
        episodic_loss = compute_episodic_loss(policy, args)
        policy.update_weights(episodic_loss)
        store.append([i_episode, total_reward])


        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d} \tTotal Reward: {}'.format(
                i_episode, t, total_reward))

            if total_reward > policy.best_policy_reward:
                policy.best_policy_reward = total_reward
                # save learnt and weights
                utils.save_weights(policy, args)
                print("Saving the best found policy")


    # save learnt rewards over episodes
    keys = ["Iteration", "Total_Reward"]
    utils.save_results(store, args, keys, session='Training')


    # test run
    if args.Test_run==True:
        utils.test(policy, args)

if __name__ == '__main__':
    REINFORCE()
