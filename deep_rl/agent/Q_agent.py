import numpy as np
from .BaseAgent import BaseAgent


class QAgent(BaseAgent):

    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.total_steps = 0
        self.lr = 0.7
        self.states = config.state_normalizer(self.task.reset())
        self.visit_states = []
        self.q_table = self._q_table()

    def step(self):
        config = self.config
        action = self.q_table[self.states].argmax()
        next_states, reward, done, info = self.task.step([action])
        reward = [-100.] if done else reward  # hack
        next_states = config.state_normalizer(next_states)
        next_action = self.q_table[next_states].argmax()
        next_q = reward + config.discount * self.q_table[next_states][next_action]
        q = self.q_table[self.states, action]
        q_value = (1 - self.lr) * q + self.lr * next_q
        self.q_table[self.states, action] = q_value
        self.states = next_states
        self.total_steps += 1

    def eval_step(self, state):
        config = self.config
        state = config.state_normalizer(state)
        action = self.q_table[state].argmax()
        return [action]

    def _q_table(self):
        config = self.config
        high = self.task.observation_space.high
        observ_size = config.state_normalizer(high) + 1
        action_size = self.task.action_space.n
        return np.random.uniform(size=(observ_size, action_size), low=-1., high=1.)
