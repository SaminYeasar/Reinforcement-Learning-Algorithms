import gym
import torch
from GAIL_mujoco import GAIL
import argparse
import numpy as np


# parser.add_argument('--max_timesteps', default=1e6, type=int)
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--policy_name', default='TD3')
	parser.add_argument('--env_name', default='HalfCheetah-v2')
	parser.add_argument('--seed', default=0, type=int)
	#parser.add_argument('--expert_traj_file',
	#					default='expert_traj/trajs/trajs_halfcheetah.h5')
	parser.add_argument('--expert_traj_file', default='../imitation/imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5')
	parser.add_argument('--start_timesteps', default=1e4, type=int)
	parser.add_argument('--eval_freq', default=5e3, type=int)
	parser.add_argument('--max_timesteps', default=1e6, type=int)
	parser.add_argument('--expl_noise', default=0.1, type=float)
	parser.add_argument('--batch_size', default=100, type=int)
	parser.add_argument('--discount', default=0.99, type=float)
	parser.add_argument('--tau', default=0.005, type=float)
	parser.add_argument('--policy_noise', default=0.2, type=float)
	parser.add_argument('--noise_clip', default=0.5, type=float)
	parser.add_argument('--entropy_lambda', default=0.1, type=float)
	parser.add_argument('--policy_freq', default=2, type=int)
	parser.add_argument('--num_traj', type=int, default=4)
	parser.add_argument('--subsamp_freq', type=int, default=20)
	parser.add_argument('--log_format', default='text', type=str)
	args = parser.parse_args()
	return args




def train():
	###################
	# define parameters
	###################
	args = parse_args()
	env = gym.make(args.env_name)

	# set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)



	#env_name = "BipedalWalker-v2"
	# env_name = "LunarLanderContinuous-v2"
	solved_reward = 3000   # for BipedalWalker-v2 it's 300
	lr = 0.0002       # learning rate  $ Make it adaptive with iteration$
	beta_init = 0.5   # parameter for Adam optimizer

	n_epoch = 25000   			 	# number of epoch
	batch_size = 100          		# number of transitions samples each time
	n_itr = 1000       	 	   		# update policy "n_itr" times at each epoch
	n_eval_trials = 100            	# evaluate learnt policy for fixed number of trials and calculate the mean rewards
	max_trajectory_steps = 1000  	# max time steps (s,a) pair taken for one epoch

	""" here Total number of iterations = epoch * n_itr"""
	###############################
	# Define environment parameters
	###############################

	#env = gym.make(env_name)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	""" don't know why we need this"""
	max_action = float(env.action_space.high[0])



	###########
	# call GAIL
	###########
	Policy = GAIL(args, state_dim, action_dim, max_action, lr, beta_init)

	###########################
	# Look for PreTrained Model
	###########################
	LookForPreTrainedModel = False
	if LookForPreTrainedModel == True:
		Policy.load()

	# Initialize variable to evaluate save weight condition
	max_avg_reward = 0

	"""number of time we do policy update"""
	for epoch in range(n_epoch):

		###########################
		# update policy n_itr times
		###########################
		""" after each n_itr we evaluate our learnt policy"""
		Policy.update(n_itr, batch_size)

		########################################
		# evaluate updated policy in environment
		########################################
		total_reward = 0

		"""use avg reward to evaluate policy over specific number of trials"""
		""" We may not want to evaluate at every epochs. Do it maybe after 100 epoch"""
		for trial in range(n_eval_trials):
			state = env.reset()
			done = False
			itr = 0
			while not done:
				""" we don't need to store state,"""
				# take action using policy
				action = Policy.select_action(state)
				# observe the next state and reward for taking action
				next_state, reward, done, _ = env.step(action.detach()) # detach from tensor

				# append reward
				total_reward += reward
				# declare state to be next_state at next iteration
				state = next_state

				# as for perfect expert it's in continuous state. Fix the number of maximum time step that it's let go in environment
				itr += 1
				if itr >= max_trajectory_steps:
					break

		####################
		# Result Calculation
		####################
		avg_reward = int(total_reward/n_eval_trials)
		print("Epoch: {}\Avg Reward: {}".format(epoch, avg_reward))

		# Save weights after number of epoch given result improvement
		# (1) Required to have improved performance
		# (2) Need to run for certain amount of epochs
		if max(max_avg_reward, avg_reward) == avg_reward and epoch % 1000 == 0:
			max_avg_reward = avg_reward
			Policy.save()

		# Condition to end training agent
		if avg_reward > solved_reward:
			print("##### solved #####")
			Policy.save()
			break


if __name__=='__main__':
	train()