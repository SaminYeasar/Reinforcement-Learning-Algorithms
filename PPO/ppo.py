import numpy as np
import torch
import torch.optim as optim
from gail.gail_models import  ActorCritic


class PPO(object):
    def __init__(self, num_inputs, num_outputs, args):
        super(PPO, self).__init__()
        # model:
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ActorCritic(num_inputs, num_outputs, args.hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[
                                                                                                           rand_ids, :]


    def update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, states, actions, log_probs,
                                                                             returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self):
        ################################################
        """ (3) Update Policy (REINFORCE is being used)"""
        ################################################
        action_log_prob_hist = trajectory['action_log_prob_hist']
        reward_hist = trajectory['reward_hist']
        # Training for the policy function (REINFORCE) (Tricks for variance reduction not implemented yet)
        policy_loss = 0.0
        returns = []
        ret = 0.0
        for log_a, r in zip(reversed(action_log_prob_hist), reversed(reward_hist)):
            ret = self.args.discount_train * ret + r
            policy_loss -= log_a * ret
        # returns.insert(0,ret)
        # returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-3)
        # policy_loss = (torch.Tensor(ac_log_prob_hist)*returns).sum()

        # update policy
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        return

