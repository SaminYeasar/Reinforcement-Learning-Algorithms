import numpy as np
import gym
from gym import wrappers

env_name = 'FrozenLake8x8-v0'
env = gym.make(env_name)
env = env.unwrapped   # source: https://stackoverflow.com/questions/52040325/openai-gym-env-p-attributeerror-timelimit-object-has-no-attribute-p

def execute_algo(env, policy, gamma = 1.0, render=False):
	total_reward = 0
	state = env.reset()
	done = False
	step_idx = 0
	while not done:
		step_idx +=1
		if render:
			env.render()

		action = policy(state)
		next_state,reward,done,_ = env.step(action)

		state = next_state
		total_reward += ((gamma**step_idx) * reward)
	return total_reward

def one_step_lookahead(env,state,gamma,V):
	Q = np.zeros(env.nA)
	for a in range(env.nA):
		for prob,next_state,reward,done in env.P[state][a]:
			Q[a] += prob * (reward + gamma * V [next_state])
	return Q

def value_iteration(env):
	print(env.P)
	v = np.zeros(env.nS)
	#for s in range (env.nS):


	return v
