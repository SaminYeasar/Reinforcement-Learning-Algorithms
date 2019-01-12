import gym
from DQN import DeepQNetwork, Agent , Replay_Buffer
import numpy as np
from utils import plotLearning


#def trunc_image(observation):
#	return np.mean(observation[15:200, 30:125], axis=2)
def preprocess(img):
	return np.mean(img[::2,::2], axis=2).astype(np.uint8)

if __name__ == '__main__':
	env = gym.make('SpaceInvaders-v0')
	print('No of possible action = {}'.format(env.action_space.n))
	# print('Dimension of observation space = {}'.format(env.observation_space.n))
	learner = Agent(env, maxMemorySize=500, alpha=0.003, epsilon=1.0, minEpsilon=0.5, gamma=0.95, replace=1000)
	buffer_ = Replay_Buffer(maxMemorySize=500)

	while buffer_.memCounter < buffer_.memSize:
		observation = env.reset()
		done = False
		while not done:
			"""
			0 = No action | 1 = fire | 2 = move right | 3 move left |
			4 = move right fire | 5 = move left fire
			
			"""
			action = env.action_space.sample()
			observation_, reward, done, _ = env.step(action)

			if done:
				reward = -100

			# store (s,a,r,s')
			#learner.storeTransition(trunc_image(observation), action, reward, trunc_image(observation_))

			transition = buffer_.storeTransition(preprocess(observation), action, reward, preprocess(observation_),done)

			# state = next_state
			observation = observation_

	# action =learner.chooseAction(env,observation)
	print('done initializing memory')

	numGames = 50
	batch_size = 32
	epsHist = []
	total_rewards = []
	for i in range(numGames):
		print('starting game {}, with epsilon {}'.format(i + 1, learner.EPSILON))
		epsHist.append(learner.EPSILON)

		observation = env.reset()
		done = False

		frames = []
		#frames.append(trunc_image(observation))
		frames.append(preprocess(observation))

		# frames = [ trunc_image(observation) ]
		rewards_per_epi = 0
		lastAction = 0

		while not done:
			if len(frames) == 3:
				action = learner.chooseAction(env, frames)
				frames = []
			else:
				action = lastAction

			observation_, reward, done, _ = env.step(action)
			rewards_per_epi += reward
			#frames.append(trunc_image(observation_))
			frames.append(preprocess(observation_))

			if done:
				reward = - 100  # otherwise gives zero; we want to give higher negative


			#learner.storeTransition(trunc_image(observation), action, reward, trunc_image(observation_))
			buffer_.storeTransition(preprocess(observation), action, reward, preprocess(observation_),done)

			observation = observation_
			sampled_memory = buffer_.sample_miniBatch(batch_size)
			learner.learn(sampled_memory)
			lastAction = action

		# env.render()
		total_rewards.append(rewards_per_epi)
		print('Total Rewards after {} episode = {}'.format(i + 1, total_rewards))
# x = [i+1 for i in range(numGames)]
# fileName = str(numGames) + 'Games' + 'Gamma' + str(learner.GAMMA) + \
#		   'Alpha' + str(learner.ALPHA) + 'Memory' + str(learner.memSize) + '.png'

# plotLearning(x,total_rewards,epsHist,fileName)
