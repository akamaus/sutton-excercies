import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F


class ScalingEncoder:
    def __init__(self, diaps):
        self.enc_size = len(diaps)
        self.inp_diap = torch.FloatTensor(diaps)
        self.theta1 = 2 / (self.inp_diap[:, 1] - self.inp_diap[:, 0])
        self.theta2 = 1 - self.theta1 * self.inp_diap[:, 1]

    def encode(self, states):
        x = torch.FloatTensor(states)
        x = self.theta1 * x + self.theta2
        return x


class SpatialEncoder:
    def __init__(self, sizes):
        self.enc_size = np.sum(sizes)
        self.sizes = sizes

    def encode(self, states):
        assert isinstance(states, list)
        x = torch.zeros([len(states), self.enc_size])

        for i, state in enumerate(states):
            p0 = 0
            for j, sj in enumerate(self.sizes):
                x[i, p0 + state[j]] = 1
                p0 += sj
        return x


class NormalizingEncoder:
    def __init__(self, sizes):
        self.enc_size = len(sizes)
        self.inp_diap = torch.FloatTensor(sizes)

    def encode(self, states):
        x = torch.FloatTensor(states)
        x = (x - self.inp_diap / 2) / self.inp_diap
        return x


class BasePolicy(nn.Module):
    """ Base policy without any input preprocessing """
    def __init__(self, n_inputs, n_actions, n_hidden):
        super().__init__()

        self.affine1 = nn.Linear(n_inputs, n_hidden)
        self.affine2 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        y = self.affine2(x)
        return y

    def select_action(self, state, t=1):
        x = self.encode_input([state])
        logits = self.forward(x)
        probs = F.softmax(logits / t, dim=1)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)[0], torch.sum(-probs * probs.log())

    def encode_input(self, states):
        assert isinstance(states, list)
        if hasattr(self, 'input_encoder'):
            return self.input_encoder.encode(states)
        else:
            return torch.FloatTensor(states)


class NormPolicy(BasePolicy):
    """ Policy centering inputs, assuming fixed discrete size of environment """
    def __init__(self, env, n_hidden=128):
        n_actions = env.N_ACTIONS
        szs = env.get_sizes()
        self.input_encoder = NormalizingEncoder(szs)
        super().__init__(n_inputs=self.input_encoder.enc_size, n_actions=n_actions, n_hidden=n_hidden)


class SpatialPolicy(BasePolicy):
    """ Policy with separate input neuron for each spatial state and axe velocity """
    def __init__(self, env, n_hidden=128):
        szs = env.get_sizes()
        n_actions = env.N_ACTIONS
        self.input_encoder = SpatialEncoder(szs)
        super().__init__(self.input_encoder.enc_size, n_actions, n_hidden=n_hidden)


class ScaledPolicy(BasePolicy):
    """ Policy with separate input neuron for each spatial state and axe velocity """
    def __init__(self, env, diaps, n_hidden=128):
        st0 = env.get_state()
        assert len(st0) == len(diaps)
        n_actions = env.N_ACTIONS
        self.input_encoder = ScalingEncoder(diaps)
        super().__init__(self.input_encoder.enc_size, n_actions, n_hidden=n_hidden)


class BaseValue(nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super().__init__()
        self.affine1 = nn.Linear(n_inputs, n_hidden)
        self.affine2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        y = self.affine2(x)[:, 0]
        return y

    def compute_value(self, states):
        if isinstance(states, tuple):
            states = [states]
            sole = True
        else:
            sole = False

        x = self.encode_value(states)
        y = self.forward(x)

        if sole:
            y = y[0]
        return y

    def encode_value(self, states):
        if hasattr(self, 'input_encoder'):
            return self.input_encoder.encode(states)
        else:
            return torch.FloatTensor(states)


class NormValue(BaseValue):
    def __init__(self, env, n_hidden=128):
        szs = env.get_sizes()
        self.input_encoder = NormalizingEncoder(szs)
        super().__init__(n_inputs=self.input_encoder.enc_size, n_hidden=n_hidden)


class SpatialValue(BaseValue):
    def __init__(self, env, n_hidden=128):
        szs = env.get_sizes()
        self.input_encoder = SpatialEncoder(szs)
        super().__init__(self.input_encoder.enc_size, n_hidden)


class ScaledValue(BaseValue):
    """ Policy with separate input neuron for each spatial state and axe velocity """
    def __init__(self, env, diaps, n_hidden=128):
        st0 = env.get_state()
        assert len(st0) == len(diaps)
        self.input_encoder = ScalingEncoder(diaps)
        super().__init__(self.input_encoder.enc_size, n_hidden=n_hidden)
