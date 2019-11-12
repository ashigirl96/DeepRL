#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision


class STACAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state, deterministic=True)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def step(self):
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = torch.tanh(action)  # squashing
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        experiences = list(zip(self.state, action, reward, next_state, done))
        self.replay.feed_batch(experiences)
        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.replay.size() >= config.warm_up:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = tensor(states)
            actions = tensor(actions)
            rewards = tensor(rewards).unsqueeze(-1)
            next_states = tensor(next_states)
            mask = tensor(1 - terminals).unsqueeze(-1)

            a_next_ = self.network(next_states)
            a_next = torch.tanh(a_next_)  # squashing

            min_a = float(self.task.action_space.low[0])
            max_a = float(self.task.action_space.high[0])
            a_next = a_next.clamp(min_a, max_a)

            q_1, q_2 = self.target_network.q(next_states, a_next)
            log_prob_1, log_prob_2 = self.network.log_prob(next_states, action_=a_next_)
            q_1 = q_1 - config.sac_coef * log_prob_1
            q_2 = q_2 - config.sac_coef * log_prob_2
            target = rewards + config.discount * mask * torch.min(q_1, q_2)
            target = target.detach()

            q_1, q_2 = self.network.q(states, actions)
            critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)
            self.logger.add_scalar('critic_loss', critic_loss)

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            if self.total_steps % config.td3_delay:
                action_ = self.network(states)
                action = torch.tanh(action_)  # squashing
                log_prob_1, log_prob_2 = self.network.log_prob(states, action_=action_)
                q_1, q_2 = self.network.q(states, action)
                policy_loss_1 = (config.sac_coef * log_prob_1 - q_1).mean()
                policy_loss_2 = (config.sac_coef * log_prob_2 - q_2).mean()
                self.logger.add_scalar('policy_loss_1', policy_loss_1)
                self.logger.add_scalar('policy_loss_2', policy_loss_2)

                self.network.zero_grad()
                policy_loss = 0.5 * (policy_loss_1 + policy_loss_2)
                self.logger.add_scalar('policy_loss', policy_loss)
                policy_loss.backward()
                self.network.actor_opt.step()

                self.soft_update(self.target_network, self.network)

            entropy = self.network.entropy()
            self.logger.add_scalar('entropy', entropy)
