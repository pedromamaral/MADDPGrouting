import torch as T
from torch import nn, tensor, rand, optim, device, cat, save, load, softmax
import torch.nn.functional as F



class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.01, beta=0.01, fc1=64, 
                    fc2=64, fa1=64, fa2=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        #self.agent_name = 'agent_joint_training_improved_strategy%s' % agent_idx
        self.agent_name = 'agent_joint_training_gcritic_updated_fixed_tm_15_02_%s' % agent_idx
        self.load_name = self.agent_name#'agent_joint_training_updated_vary_tm_fix_16_11_8.0%s' % agent_idx
        self.actor = ActorNetwork(alpha, actor_dims, fa1, fa2, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor', load_file=self.load_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic', load_file=self.load_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fa1, fa2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor', load_file=self.load_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic', load_file=self.load_name+'_target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)
        noise = T.rand(self.n_actions).to(self.actor.device)
        action = actions
        return action.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 n_agents, n_actions, name, chkpt_dir, load_file):
        super(CriticNetwork, self).__init__()

        self.file_name = f"{name}.sync"  # os.path.join(chkpt_dir, name)
        self.chkpt_file = f'/content/drive/MyDrive/{self.file_name}'
        self.load_file = f'/content/drive/MyDrive/{load_file}.sync'

        self.fc1 = nn.Linear(input_dims + n_actions, fc1_dims).float()
        # self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc1_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        # x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.load_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir, load_file):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = f"{name}.sync"  # os.path.join(chkpt_dir, name)
        self.chkpt_file = f'/content/drive/MyDrive/{self.chkpt_file}'
        self.load_file = f'/content/drive/MyDrive/{load_file}.sync'

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        # self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc1_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        pi = T.softmax(self.pi(x), dim=1)

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.load_file))