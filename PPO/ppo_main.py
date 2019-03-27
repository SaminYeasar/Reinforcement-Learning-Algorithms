import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import argparse
from gail.models import ActorCritic
from gail.ppo import PPO
# arguments
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="PPO", type=str)
	parser.add_argument("--hidden_size", default=256, type=int)
	parser.add_argument("--discrim_hidden_size", default=128, type=int)
	parser.add_argument("--learning_rate", default=3e-4, type=int)
	parser.add_argument("--num_steps", default=20, type=int)
	parser.add_argument("--mini_batch_size", default=5, type=int)
	parser.add_argument("--ppo_epochs", default=4, type=int)
	parser.add_argument("--threshold_reward", default=-200, type=int)
	parser.add_argument("--max_frame", default=15000, type=int)
	args = parser.parse_args()
	return args

# initialize arg:
args = parse_args()

# cuda:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# environment:
env_name = "Pendulum-v0"
env = gym.make(env_name)
num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]

if args.policy == "PPO":
	agent = PPO(num_inputs, num_outputs, args)

def test_env(vis=False):
	state = env.reset()
	if vis:
		env.render()
	done = False
	total_reward = 0
	while not done:
		state = torch.FloatTensor(state).unsqueeze(0).to(device)
		dist, _ = agent.model(state)
		next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
		state = next_state
		if vis: env.render()
		total_reward += reward
	return total_reward

def plot(frame_idx, rewards):
	#clear_output(True)
	plt.figure(figsize=(20, 5))
	plt.subplot(131)
	plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
	plt.plot(rewards)
	plt.show()

frame_idx  = 0
test_rewards = []



state = env.reset()
early_stop = False

while frame_idx < args.max_frame:

	log_probs = []
	values = []
	states = []
	actions = []
	rewards = []
	masks = []
	entropy = 0

	# for AIS uses 300 steps
	for _ in range(args.num_steps):
		state = torch.FloatTensor(state).to(device)
		dist, value = agent.model(state)

		action = dist.sample()
		next_state, reward, done, _ = env.step(action.cpu().numpy())

		log_prob = dist.log_prob(action)
		entropy += dist.entropy().mean()

		log_probs.append(log_prob.unsqueeze(0))
		values.append(value)
		rewards.append(torch.FloatTensor(np.array(reward)).view(-1,1).to(device))
		masks.append(torch.FloatTensor(np.array(reward)).view(-1,1).to(device))

		states.append(torch.FloatTensor(np.array(state)).unsqueeze(0))
		actions.append(torch.FloatTensor(np.array(action)).unsqueeze(0))

		state = next_state
		frame_idx += 1

		if frame_idx % 1000 == 0:
			test_reward = np.mean([test_env() for _ in range(10)])
			test_rewards.append(test_reward)
			plot(frame_idx, test_rewards)
			if test_reward > args.threshold_reward: early_stop = True

	next_state = torch.FloatTensor(next_state).to(device)
	_, next_value = agent.model(next_state)
	returns = agent.compute_gae(next_value, rewards, masks, values)

	returns = torch.cat(returns).detach()
	log_probs = torch.cat(log_probs).detach()
	values = torch.cat(values).detach()
	states = torch.cat(states)
	actions = torch.cat(actions)
	advantage = returns - values

	agent.update(args.ppo_epochs, args.mini_batch_size, states, actions, log_probs, returns, advantage)



