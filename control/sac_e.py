"""
soft actor critic implementation
"""
import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
from control.networks import QNet,  PolicyNet,  soft_update
from control.memory import ReplayBuffer


class SoftActorCritic(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim,
                 device,
                 lr=1e-3,
                 tau=0.005,
                 mem_capacity=50000,
                 mem_batch_size=256,
                 mem_batch_abs_mode=True,
                 mem_min_samples=1000,
                 mem_update_interval=1,
                 alpha=0.01,
                 gamma=0.99,
                 ):

        self.alpha = alpha
        self.device = device
        self.gamma = gamma
        self.tau = tau

        # setup replay buffer
        self.replay_buffer = ReplayBuffer(self.device, mem_capacity, mem_batch_size, state_dim, action_dim,
                                          mem_batch_abs_mode, mem_min_samples, mem_update_interval)

        # declare policy network
        self.policy = PolicyNet(state_dim, action_dim, hidden_dim).to(device)

        # declare q networks
        self.q1 = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.q2 = QNet(state_dim, action_dim, hidden_dim).to(device)

        # declare target q networks
        self.q1_target = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.q2_target = QNet(state_dim, action_dim, hidden_dim).to(device)

        # copy weights from q to q target networks ( soft tau = 1 -> copy)
        soft_update(self.q1_target, self.q1, tau=1)
        soft_update(self.q2_target, self.q2, tau=1)

        # entropy tuning
        self.target_entropy = -torch.prod(torch.Tensor([action_dim, ]).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        # loss functions
        self.q1_loss_f = nn.MSELoss()
        self.q2_loss_f = nn.MSELoss()

        # optimizers
        self.q1_optimizer = opt.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = opt.Adam(self.q2.parameters(), lr=lr)
        self.policy_optimizer = opt.Adam(self.policy.parameters(), lr=lr)
        self.alpha_optimizer = opt.Adam([self.log_alpha], lr=lr)

    def get_action(self, s):
        state = torch.FloatTensor(s).to(self.device).unsqueeze(0)
        action = self.policy.get_action(state)
        #return action.detach()[0].cpu().detach()

        return action[0].cpu().detach()

    def update(self):
        # get random training samples
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # compute target outputs
        with torch.no_grad():
            next_action, next_log_pi = self.policy.get_action_logprob(next_states)
            next_target_q = torch.min(self.q1_target(next_states, next_action),
                                      self.q2_target(next_states, next_action))
            next_target_q -= (self.alpha * next_log_pi)
            next_q = rewards + self.gamma * (1 - dones) * next_target_q

        # optimize q1 and soft update q1 target network
        q1_loss = self.q1_loss_f(self.q1(states, actions), next_q)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        soft_update(self.q1_target, self.q1, tau=self.tau)

        # optimize q2 and soft update q2 target network
        q2_loss = self.q2_loss_f(self.q2(states, actions), next_q)
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        soft_update(self.q2_target, self.q2, tau=self.tau)

        # optimize policy network
        pi, log_pi = self.policy.get_action_logprob(states)
        min_q_pi = torch.min(self.q1(states, pi), self.q2(states, pi))
        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # optimize entropy
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

    def push_buffer(self, state, action, reward, next_state, done):

        # convert all training variables to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.from_numpy(np.array([reward])).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.from_numpy(np.array([float(done)])).to(self.device)

        # send data to replay buffer, returns true if there is more than N new elements in memory
        return self.replay_buffer.push(state, action, reward, next_state, done)
