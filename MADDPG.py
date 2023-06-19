import random

import torch as T
import numpy as np
from torch import tensor, cat, no_grad, mean
import torch.nn.functional as F

from Agent import Agent
from MultiAgentReplayBuffer import MultiAgentReplayBuffer
from NetworkEngine import NetworkEngine
from NetworkEnv import NetworkEnv
from environmental_variables import STATE_SIZE, EPOCH_SIZE, NUMBER_OF_AGENTS


class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 scenario='simple', alpha=0.01, beta=0.01, fc1=64,
                 fc2=64, fa1=64, fa2=64, gamma=0.99, tau=0.001, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims[agent_idx],
                                     n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                                     chkpt_dir=chkpt_dir, fc1=fc1, fc2=fc2))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(np.argmax(action))
        return actions

    def learn(self, experience):
        if not experience.ready():
            return
        actor_input, current_state, action_taken, reward_obtained, \
        actor_future_input, future_state, done_flags = experience.sample_buffer()

        processing_device = self.agents[0].actor.device
        current_state = T.tensor(current_state, dtype=T.float).to(processing_device)
        action_taken = T.tensor(action_taken, dtype=T.float).to(processing_device)
        reward_obtained = T.tensor(reward_obtained, dtype=T.float).to(processing_device)
        future_state = T.tensor(future_state, dtype=T.float).to(processing_device)
        done_flags = T.tensor(done_flags).to(processing_device)

        all_new_actions = []
        previous_actions = []

        for idx, agent in enumerate(self.agents):
            future_actor_input = T.tensor(actor_future_input[idx],
                                          dtype=T.float).to(processing_device)

            new_action_policy = agent.target_actor.forward(future_actor_input)

            all_new_actions.append(new_action_policy)
            previous_actions.append(action_taken[idx])

        combined_new_actions = T.cat([act for act in all_new_actions], dim=1)
        combined_old_actions = T.cat([act for act in previous_actions], dim=1)

        for idx, agent in enumerate(self.agents):
            with T.no_grad():
                future_critic_value = agent.target_critic.forward(future_state[idx], combined_new_actions[:,
                                                                                     idx * self.n_actions:idx * self.n_actions + self.n_actions]).flatten()
                expected_value = reward_obtained[:, idx] + (
                            1 - done_flags[:, 0].int()) * agent.gamma * future_critic_value

            present_critic_value = agent.critic.forward(current_state[idx], combined_old_actions[:,
                                                                            idx * self.n_actions:idx * self.n_actions + self.n_actions]).flatten()

            critic_loss = F.mse_loss(expected_value, present_critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            current_actor_input = T.tensor(actor_input[idx], dtype=T.float).to(processing_device)
            combined_old_actions_clone = combined_old_actions.clone()
            combined_old_actions_clone[:,
            idx * self.n_actions:idx * self.n_actions + self.n_actions] = agent.actor.forward(
                current_actor_input)
            actor_loss = -T.mean(agent.critic.forward(current_state[idx], combined_old_actions_clone[:,
                                                                          idx * self.n_actions:idx * self.n_actions + self.n_actions]).flatten())
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

        for agent in self.agents:
            agent.update_network_parameters()


if __name__ == '__main__':

    row = 9324
    col = 1
    UPDATE_STEPS = 16
    eng = NetworkEngine()

    env = NetworkEnv(eng)

    n_state = 845
    n_action = 3

    # onlineQNetwork = QNetwork()
    total_rewards = []
    total_rewards_og = []
    total_rewards_new = []

    agents = eng.get_all_hosts()

    all_hosts = eng.get_all_hosts()

    agent_dims = [STATE_SIZE for host in all_hosts]

    GAMMA = 0.99
    EXPLORE = 20000
    INITIAL_EPSILON = 0.5
    FINAL_EPSILON = 0.0001
    REPLAY_MEMORY = 50000
    BATCH = 256

    agent_dim = STATE_SIZE

    critic_dim = len(eng.get_link_usage()) + NUMBER_OF_AGENTS
    critic_dims = [critic_dim for i in range(NUMBER_OF_AGENTS)]

    maddpg_agents = MADDPG(agent_dims, critic_dims, NUMBER_OF_AGENTS, n_action,
                           fa1=10, fa2=80, fc1=15, fc2=80,
                           alpha=0.0001, beta=0.0001, tau=0.0001,
                           chkpt_dir='.\\tmp\\maddpg\\')

    memory = MultiAgentReplayBuffer(1000, critic_dims, agent_dims,
                                    n_action, NUMBER_OF_AGENTS, batch_size=100)

    evaluate = False

    if evaluate:
        maddpg_agents.load_checkpoint()

    all_rewards = []

    evaluate = False
    print(all_hosts[10])
    nr_trains = 1

    nr_epochs = 10000 if not evaluate else 1

    for epoch in range(0, nr_epochs):
        total_epoch_reward = 0
        total_epoch_pck_loss = 0
        print("Epoch: ", epoch)

        episode_size = EPOCH_SIZE if not evaluate else EPOCH_SIZE * 2
        for e in range(episode_size):

            new_tm = e % 2 == 0
            env.reset(new_tm)

            episode_reward = 0

            total_reward = 0
            total_package_loss = 0
            for time_steps in range(100):
                actions = {}

                prev_states = {}

                next_dsts = eng.get_nexts_dsts()

                # print("next dsts", next_dsts)

                all_dsts = []
                for host in all_hosts:
                    if host in next_dsts and next_dsts[host]:
                        d = next_dsts[host][1:]
                        all_dsts.append(d)
                    else:
                        all_dsts.append(0)

                states = []  # np.empty((50, agent_dim), dtype=np.double)
                critic_states = []

                dismiss_indexes = []

                for index, host in enumerate(all_hosts):
                    all_dst_states = eng.get_state(host, 1)
                    dst = next_dsts.get(host, '')
                    dst = '' if dst == None else dst

                    if 'H' not in dst:
                        state = np.zeros((1, agent_dims[index]), dtype=np.double)
                        dismiss_indexes.append(index)
                    else:
                        state = all_dst_states
                        base_state = all_dst_states

                    states.append(state)
                    critic_states.append(np.concatenate((eng.get_link_usage(), np.array(all_dsts)), axis=0))

                actions = maddpg_agents.choose_action(states)

                actions_dict = {}
                for index, host in enumerate(all_hosts):
                    if next_dsts.get(host, ''):

                        prob = -1 if evaluate else max(0.1, (0.3 - 0.0001 * epoch))

                        if random.random() < prob:
                            action = random.randint(0, 2)
                        else:
                            action = actions[index]

                        if host in eng.single_con_hosts:
                            action = 0

                        actions_dict[host] = {next_dsts.get(host, ''): action}

                next_states, rewards, done, _ = env.step(actions_dict)

                new_next_states = np.empty((25, agent_dim), dtype=np.double)

                all_critic_new_states = [np.concatenate((eng.get_link_usage(), np.array(all_dsts)), axis=0) for i in
                                         range(NUMBER_OF_AGENTS)]

                new_next_states = []
                for index, host in enumerate(all_hosts):
                    # means it add an action
                    if host in actions_dict and next_dsts[host]:
                        bw_state = next_states[host]
                        new_next_states.append(bw_state)
                    else:
                        new_next_states.append(np.zeros((1, agent_dims[index]), dtype=np.double))

                actions = []

                for host in all_hosts:
                    if host not in actions_dict:
                        actions.append(0)
                    else:
                        actions.append(actions_dict[host][next_dsts[host]])

                memory.store_transition(states, actions, rewards, new_next_states, done, critic_states,
                                        all_critic_new_states)

                learn_steps = 0

                total_reward += sum(rewards) / 25
                total_package_loss += eng.statistics['package_loss']
                if done:
                    break

            print("Total reward", total_reward)
            print("Total package loss", total_package_loss)

            if e % 3 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            total_epoch_reward += total_reward
            total_epoch_pck_loss += eng.statistics['package_loss']
            # print(f"STATISTICS OG {eng.statistics}")

            total_rewards.append(total_reward)
            all_rewards.append(total_reward)

            # print(f"{'OG' if epoch % 2 == 0 else 'NEW'} REWARD {total_reward}")

        print(f"total epoch reward {total_epoch_reward}")
        # f.write(f"{epoch} {total_epoch_reward}\n")


        if epoch % 30 == 0:
            print(f"AVERGAE WAS {sum(total_rewards) / len(total_rewards)}")
            total_rewards = []

            if not evaluate:
                maddpg_agents.save_checkpoint()
                print("SAVING")

        print(total_epoch_pck_loss)