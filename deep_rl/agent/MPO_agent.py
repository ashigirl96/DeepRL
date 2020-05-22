from ..network import *
from ..component import *
from .BaseAgent import *


class MPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.actor_opt = config.actor_opt_fn(self.network.actor_params)
        self.critic_opt = config.critic_opt_fn(self.network.critic_params)
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

    def step(self):
        config = self.config
        eta = config.mpo_eta
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            # prediction = {'a': action, 'log_pi_a': log_prob, 'ent': entropy, 'mean': mean, 'v': v}
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         's': tensor(states)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))  # [1, 1]
        returns = prediction['v'].detach()  # [1, 1]
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        states, actions, log_probs_old, returns, advantages = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
        actions = actions.detach()  # [2048, 2]
        log_probs_old = log_probs_old.detach()  # [2048, 1]
        advantages = (advantages - advantages.mean()) / advantages.std()  # [2048, 1]

        # loop 640
        for _ in range(config.optimization_epochs):  # 10
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:  # mini_batch_size: 64
                batch_indices = tensor(batch_indices).long()  # [64]
                sampled_states = states[batch_indices]  # [64, 9]
                sampled_actions = actions[batch_indices]  # [64, 2]
                sampled_log_probs_old = log_probs_old[batch_indices]  # [64, 1]
                sampled_returns = returns[batch_indices]  # [64, 1]
                sampled_advantages = advantages[batch_indices]  # [64, 1]

                prediction = self.network(sampled_states, sampled_actions)
                # TODO: Compute gradients for V
                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()
                # TODO: Compute gradients for \eta
                def dual(eta):
                    baseline_advantages_ = torch.max(sampled_advantages, 0)[0]  # []
                # TODO: Compute gradients for \pi
                baseline_advantages_ = torch.max(sampled_advantages, 0)[0]  # []
                exp_advantages = torch.exp((sampled_advantages - baseline_advantages_) / eta)
                normalization = torch.mean(exp_advantages, 0)
                baseline_advantages = exp_advantages / normalization




                # exp_adv * prediction['log_pi_a']
                # TODO: Compute gradients for \eta

                # ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()  # [64, 1]
                # obj = ratio * sampled_advantages  # [64, 1]
                # obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                #                           1.0 + self.config.ppo_ratio_clip) * sampled_advantages  # [64, 1]
                # policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()  # []
                # value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean() # []
                #
                # approx_kl = (sampled_log_probs_old - prediction['log_pi_a']).mean()  # []
                # if approx_kl <= 1.5 * config.target_kl:
                #     self.actor_opt.zero_grad()
                #     policy_loss.backward()
                #     self.actor_opt.step()
                #
                # self.critic_opt.zero_grad()
                # value_loss.backward()
                # self.critic_opt.step()


