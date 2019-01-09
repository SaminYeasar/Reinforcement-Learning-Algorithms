import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
	def __init__(self, ALPHA):
		super(DeepQNetwork, self).__init__()

		# construct NN layers
		#self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
		self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		#self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		#self.fc1 = nn.Linear(128 * 19 * 8, 512)
		self.fc1 = nn.Linear( 64*9*6, 512)  # no of channels * width * height
		self.fc2 = nn.Linear(512, 6)

		self.relu = nn.ReLU()

		# optimizer
		self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)

		# compute loss
		self.loss = nn.MSELoss()

		# torch stuff
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		# send the network to device
		self.to(self.device)
	"""
	def forward(self, observation):
		# convert observation to tensor
		# send variable to device
		observation = torch.Tensor(observation).to(self.device)
		# crop shape
		observation = observation.view(-1, 1, 185,
									   95)  # (-1 so that can handle any number of samples, no of channel , frame X, frame Y)

		# feed observation to NN
		observation = F.relu(self.conv1(observation))
		observation = F.relu(self.conv2(observation))
		observation = F.relu(self.conv3(observation))
		observation = observation.view(-1, 128 * 19 * 8)
		observation = F.relu(self.fc1(observation))
		action = self.fc2(observation)  # action is a matrix
		return action
	"""

	def forward(self, obs):
		"""
		obs(observation) = (num of samples , width , height , num of input channel = 1)
		"""
		obs = torch.Tensor(obs).to(self.device)
		obs = obs.view(-1, 1, obs.shape[1],obs.shape[2]) # 32,number of channels = 1 , 105, 80
		obs = self.relu(self.conv1(obs))  # (num of samples , updated width , updated height , num of input channel = 32)
		obs = self.relu(self.conv2(obs))  # (num of samples , updated width , updated height , num of input channel = 64)
		obs = self.relu(self.conv3(obs))  # (num of samples , updated width , updated height , num of input channel = 64)

		# x.size(0) =  number of samples
		obs = obs.view(obs.size(0), -1)  # (num of samples , updated width * updated height * num of input channel = 64)
		obs = self.relu(self.fc1(obs))  # (num of samples , 512)
		action = self.fc2(obs)  # (num of samples, possible actions)
		return action

#####################################################################


class Agent(object):
	def __init__(self, env, maxMemorySize, alpha, epsilon, minEpsilon, gamma, replace=None):
		"""
		gamma = discount factor
		alpha = agent learning rate
		epsilon = with probability agent takes random action
		"""
		self.ALPHA = alpha
		# super(DeepQNetwork, self).__init__()
		#######################################################
		# construct memory (will be used in "storeTransition")
		self.memSize = maxMemorySize
		self.memory = []
		self.memCounter = 0

		#######################################################
		# (will be used in "chooseAction")
		self.Q_eval = DeepQNetwork(alpha)  # feeding the network
		# define the start and end of epsilon updates
		# decrease epsilon after fixed number of episodes (ex: 500 used in here)
		self.EPSILON = epsilon
		self.minEPSILON = minEpsilon
		self.actionSpace = env.action_space.n
		self.steps = 0  # keep track number of times action os taken

		#######################################################
		# (will be used in "learn")
		self.Q_next = DeepQNetwork(alpha)  # feeding the network
		self.GAMMA = gamma
		self.learn_step_counter = 0  # keep track number of times "learn" executed

	def storeTransition(self, state, action, reward, state_):

		if self.memCounter < self.memSize:
			self.memory.append([state, action, reward, state_])
		else:
			# 10%100 = 10; 35%100=35; 100%100=0; 105%100=5; 113%100=100
			self.memory[self.memCounter % self.memSize] = [state, action, reward, state_]
		self.memCounter += 1

	def chooseAction(self, env, observation):
		rand = np.random.random()

		# get action(matrix) for given observation using NN
		"""action = probability matrix of possible actions
			        action 1 | action 2| action 3| ..
			frame 1|
			frame 2|
			frame 3|
			 .
			 .
		"""
		action = self.Q_eval.forward(observation)

		# implement simple epsilon-greedy
		if rand < 1 - self.EPSILON:
			action = torch.argmax(action[1]).item()
		else:
			action = np.random.choice(self.actionSpace)

		self.steps += 1  # keep track number of times we have taken action
		return action

	def learn(self, batch_size):
		# we use batch sampling to learn so that
		#	 	- we don't get stuck at local optimia
		#		- or don't constantly sample from co-related sample

		# clear out the gradients of all Variables every-time we do backprop to make sure it's a pure "batch learning"
		# IMPORTANT:https://stackoverflow.com/questions/48001598/why-is-zero-grad-needed-for-optimization
		self.Q_eval.optimizer.zero_grad()

		###########################
		# The target network has its weights kept frozen most of the time,
		# but is updated with the policy network’s weights every so often
		# NOTE: not using it in the code so "replace_target_counter" is not defined.
		"""
		if self.replace_target_counter is not None and \
				self.learn_step_counter % self.replace_target_counter == 0:
			self.Q_next.load_state_dict(self.Q_eval.state_dict())  # load_state_dict is pytorch command
		"""
		##########################

		# sample Mini-Batch
		if self.memCounter + batch_size < self.memSize:
			memStart = int(np.random.choice(range(self.memCounter)))
		else:
			memStart = int(np.random.choice(range(self.memSize - batch_size - 1)))

		miniBatch = self.memory[memStart:memStart + batch_size]
		memory = np.array(miniBatch)

		# memory = sample_miniBatch(self, batch_size)

		# convert numpy array to list as pytorch doesn't take numpy array
		# memory[:,0][:] all rows(entire batch) , zero'th element(state) and all of the pixels
		# send it to device
		# (state,action,reward,state_)

		###################
		# Feed-forward :
		####################
		Qpred = self.Q_eval.forward(list(memory[:, 0][:])).to(self.Q_eval.device)
		Qnext = self.Q_next.forward(list(memory[:, 3][:])).to(self.Q_eval.device)

		# output of Qnext if probaility matrix of actions, where column is different actions for different batch along the row
		maxA = torch.argmax(Qnext, dim=1).to(self.Q_eval.device)
		reward = torch.Tensor(list(memory[:, 2])).to(self.Q_eval.device)
		Qtarget = Qpred
		Qtarget[:, maxA] = reward + self.GAMMA * torch.max(Qnext[1])

		# Update epsilon
		if self.steps > 500:
			if self.EPSILON - 1e-4 > self.minEPSILON:
				self.EPSILON -= 1e-4
			else:
				self.EPSILON = self.minEPSILON

		# self.update_epsilon()

		##################
		# Compute loss:
		##################
		loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)

		###################
		# Back-propagation:
		###################
		loss.backward()

		###################
		# Update network parameter
		###################
		self.Q_eval.optimizer.step()  # torch stuff

		self.learn_step_counter += 1

	def update_epsilon(self):
		if self.steps > 500:
			if self.EPSILON - 1e-4 > self.minEPSILON:
				self.EPSILON -= 1e-4
			else:
				self.EPSILON = self.minEPSILON

	def sample_miniBatch(self, batch_size):
		# Sample Mini-Batch
		if self.memCounter + batch_size < self.memSize:
			memStart = int(np.random.choice(range(self.memCounter)))
		else:
			memStart = int(np.random.choice(range(self.memCounter - batch_size - 1)))

		miniBatch = self.memory[memStart:memStart + batch_size]
		memory = np.array(miniBatch)
		return memory
