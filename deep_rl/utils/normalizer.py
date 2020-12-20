#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
import torch
from baselines.common.running_mean_std import RunningMeanStd


class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return


class MeanStdNormalizer(BaseNormalizer):
    def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):
        BaseNormalizer.__init__(self, read_only)
        self.read_only = read_only
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        x = np.asarray(x)
        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        if not self.read_only:
            self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                       -self.clip, self.clip)

    def state_dict(self):
        return {'mean': self.rms.mean,
                'var': self.rms.var}

    def load_state_dict(self, saved):
        self.rms.mean = saved['mean']
        self.rms.var = saved['var']


class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)
        self.coef = coef

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return self.coef * x


class ImageNormalizer(RescaleNormalizer):
    def __init__(self):
        RescaleNormalizer.__init__(self, 1.0 / 255)


class SignNormalizer(BaseNormalizer):
    def __call__(self, x):
        return np.sign(x)


class DiscrezeNormalizer(BaseNormalizer):
    def __init__(self, observation_space, discrete=6):
        BaseNormalizer.__init__(self, False)
        self.observation_space = observation_space
        self.discrete = discrete

    def __call__(self, x):
        x = np.asarray(x)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        bins = [np.linspace(low, high, self.discrete + 1)[1:-1] for low, high in
                zip(self.observation_space.low, self.observation_space.high)]
        digitize_x = [np.digitize(o, bins=bin_) for o, bin_ in zip(x, bins)]
        int_x = np.sum([observation * (self.discrete ** i) for i, observation in
                             enumerate(digitize_x)])
        return int_x
