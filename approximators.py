import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F

EPS = 1e-9

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
        offsets = []
        off = 0
        for s in sizes:
            offsets.append(off)
            off += s
        self.offsets = torch.tensor(offsets)

    def encode(self, states):
        assert not isinstance(states, tuple)
        states_t = torch.tensor(states) + self.offsets
        x = torch.zeros(*states_t.shape[:-1], self.enc_size)

        x.scatter_(-1, states_t, 1)
        return x


class QuantizingEncoder(SpatialEncoder):
    """ Encoder which quantizes input features and makes spatial encoding out of them """
    def __init__(self, diaps, levels):
        super().__init__(levels)
        self.inp_diap = torch.FloatTensor(diaps)
        self.levels = torch.tensor(levels).float() - 1
        self.theta1 = self.levels / (self.inp_diap[:, 1] - self.inp_diap[:, 0])
        self.theta2 = - self.theta1 * self.inp_diap[:, 0]

    def encode(self, states):
        assert isinstance(states, list)
        x = torch.FloatTensor(states)
        x = self.theta1 * x + self.theta2
        x = torch.max(x, torch.tensor(0.0))
        x = torch.min(self.levels, x)
        x = x.round().long()

        return super().encode(x)

class NormalizingEncoder:
    def __init__(self, sizes):
        self.enc_size = len(sizes)
        self.inp_diap = torch.FloatTensor(sizes)

    def encode(self, states):
        x = torch.FloatTensor(states)
        x = (x - self.inp_diap / 2) / self.inp_diap
        return x


class Approximator(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden, num_layers=2):
        super().__init__()
        assert num_layers >= 2
        layers = []
        layers.append(nn.Linear(n_inputs, n_hidden))
        for hi in range(2, num_layers):
            layers.append(nn.Linear(n_hidden, n_hidden))
        layers.append(nn.Linear(n_hidden, n_outputs))
        self.layers = nn.ModuleList(layers)
        self.device = None

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        y = self.layers[-1](x)
        return y

    def encode_input(self, states):
        assert isinstance(states, list)
        if hasattr(self, 'input_encoder'):
            return self.input_encoder.encode(states)
        else:
            return torch.FloatTensor(states)


class BasePolicy(Approximator):
    """ Base policy without any input preprocessing """

    def select_action(self, state, t=1):
        if isinstance(state, tuple):
            wrapped = True
            state = [state]
        else:
            wrapped = False
        x = self.encode_input(state).to(self.device)
        logits = self.forward(x)
        probs = F.softmax(logits / t, dim=1)
        m = Categorical(logits=logits / t)
        action = m.sample()
        if wrapped:
            lp = m.log_prob(action)[0]
            action = action.item()
        else:
            action = action.detach()
            lp = m.log_prob(action)
            action = action.cpu()
        return action, lp, torch.sum(-probs * (probs.log() + EPS)) / len(x)


class NormPolicy(BasePolicy):
    """ Policy centering inputs, assuming fixed discrete size of environment """
    def __init__(self, env, **kargs):
        n_actions = env.N_ACTIONS
        szs = env.get_sizes()
        self.input_encoder = NormalizingEncoder(szs)
        super().__init__(n_inputs=self.input_encoder.enc_size, n_actions=n_actions, **kargs)


class SpatialPolicy(BasePolicy):
    """ Policy with separate input neuron for each spatial state and axe velocity """
    def __init__(self, env, **kargs):
        szs = env.get_sizes()
        n_actions = env.N_ACTIONS
        self.input_encoder = SpatialEncoder(szs)
        super().__init__(self.input_encoder.enc_size, n_actions, **kargs)


class ScaledPolicy(BasePolicy):
    """ Policy with neuron input scaled by normalizing diapasons to [-1,1] """
    def __init__(self, env, diaps, **kargs):
        st0 = env.get_state()
        assert len(st0) == len(diaps)
        n_actions = env.N_ACTIONS
        self.input_encoder = ScalingEncoder(diaps)
        super().__init__(self.input_encoder.enc_size, n_actions, **kargs)


class QuantizedPolicy(BasePolicy):
    """ Policy with separate input neuron for each quantized level of state feature """
    def __init__(self, env, diaps, levels, **kargs):
        st0 = env.get_state()
        assert len(st0) == len(diaps)
        n_actions = env.N_ACTIONS
        self.input_encoder = QuantizingEncoder(diaps, levels)
        super().__init__(self.input_encoder.enc_size, n_actions, **kargs)


class BaseValue(Approximator):
    def __init__(self, n_inputs, n_hidden, **kargs):
        super().__init__(n_inputs, 1, n_hidden, **kargs)

    def compute_value(self, states):
        if isinstance(states, tuple):
            states = [states]
            sole = True
        else:
            sole = False

        x = self.encode_value(states).to(self.device)
        y = self.forward(x).squeeze(-1)

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
    """ Value function with normalized feature input"""
    def __init__(self, env, diaps, n_hidden=128):
        st0 = env.get_state()
        assert len(st0) == len(diaps)
        self.input_encoder = ScalingEncoder(diaps)
        super().__init__(self.input_encoder.enc_size, n_hidden=n_hidden)


class QuantizedValue(BaseValue):
    """ Value function with separate input neuron for each spatial state and axe velocity """
    def __init__(self, env, diaps, levels, **kargs):
        st0 = env.get_state()
        assert len(st0) == len(diaps)
        self.input_encoder = QuantizingEncoder(diaps, levels)
        super().__init__(self.input_encoder.enc_size, **kargs)
