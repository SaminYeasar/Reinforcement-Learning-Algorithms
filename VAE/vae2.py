import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import utils
import gym

parser = argparse.ArgumentParser(description='VAE learning expert (s,a) distribution')
parser.add_argument('--env_name', type=str, default='Ant-v2')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=4, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--expert_traj_file', default='../imitation/imitation_runs/modern_stochastic/trajs/trajs_ant.h5',
					help='../imitation/imitation_runs/modern_stochastic/trajs/trajs_halfcheetah.h5')
parser.add_argument('--num_traj', type=int, default=10)
parser.add_argument('--subsamp_freq', type=int, default=20)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--max_iterations', default=1000, type=int)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(VAE, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # encoder
        self.fc1 = nn.Linear(state_dim+action_dim, 400)   # encode layer
        self.fc21 = nn.Linear(400, 20)				      # mu
        self.fc22 = nn.Linear(400, 20)					  # log variance
        # decoder
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, state_dim+action_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, xu):
        mu, logvar = self.encode(xu)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar





# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, state_action, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, state_action, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(model, epoch):
    expert_buffer = utils.ExpertBuffer(args.expert_traj_file, args.num_traj, args.subsamp_freq)
    model.train()
    train_loss = 0
    for itr in range(args.max_iterations):
        state, _, action, _, _ = expert_buffer.sample(args.batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        state_action = torch.cat([state, action], dim=1)


        optimizer.zero_grad()
        recon_batch, mu, logvar = model(state_action)
        loss = loss_function(recon_batch, state_action, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if itr % args.log_interval == 0:
            print('Train Epoch: {} | Iteration {}\tLoss: {:.6f}'.format(
                epoch, itr, loss.item() / args.batch_size))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / (args.batch_size*itr)  ))


def test(model, epoch):
    expert_buffer = utils.ExpertBuffer(args.expert_traj_file, args.num_traj, args.subsamp_freq)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for itr in range(100):
            state, _, action, _, _ = expert_buffer.sample(args.batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            state_action = torch.cat([state, action], dim=1)

            recon_batch, mu, logvar = model(state_action)
            test_loss += loss_function(recon_batch, state_action, mu, logvar).item()

    test_loss /= 100*args.batch_size
    print('====> Epoch:{} Test set loss: {:.4f}'.format(epoch, test_loss))

if __name__ == "__main__":
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = VAE(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, args.epochs + 1):

        train(model, epoch)
        test(model, epoch)
        utils.save_weights(model,args)
