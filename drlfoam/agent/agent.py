
from typing import Callable
from abc import ABC, abstractmethod, abstractproperty
import torch as pt
from ..constants import DEFAULT_TENSOR_TYPE


pt.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


def compute_returns(rewards: pt.Tensor, gamma: float = 0.99) -> pt.Tensor:
    n_steps = len(rewards)
    discounts = pt.logspace(0, n_steps-1, n_steps, gamma)
    returns = [(discounts[:n_steps-t] * rewards[t:]).sum()
               for t in range(n_steps)]
    return pt.tensor(returns)


def compute_gae(rewards: pt.Tensor, values: pt.Tensor, gamma: float = 0.99, lam: float = 0.97) -> pt.Tensor:
    n_steps = len(rewards)
    factor = pt.logspace(0, n_steps-1, n_steps, gamma*lam)
    delta = rewards[:-1] + gamma * values[1:] - values[:-1]
    gae = [(factor[:n_steps-t-1] * delta[t:]).sum()
           for t in range(n_steps - 1)]
    return pt.tensor(gae)


class FCPolicy(pt.nn.Module):
    def __init__(self, n_states: int, n_actions: int, action_min: pt.Tensor,
                 action_max: pt.Tensor, n_output: int, n_layers: int = 2, n_neurons: int = 64,
                 activation: Callable = pt.nn.functional.relu):
        super(FCPolicy, self).__init__()
        self._n_states = n_states
        self._n_actions = n_actions
        self._action_min = action_min
        self._action_max = action_max
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation
        self._n_output = n_output           # number of output neurons, for smoother = e.g. amount of available smoother

        # set up policy network
        self._layers = pt.nn.ModuleList()
        self._layers.append(pt.nn.Linear(self._n_states, self._n_neurons))
        if self._n_layers > 1:
            for hidden in range(self._n_layers - 1):
                self._layers.append(pt.nn.Linear(self._n_neurons, self._n_neurons))
                self._layers.append(pt.nn.LayerNorm(self._n_neurons))
        # smoother: 1 action -> selection of smoother, but 6 smoother available -> 6 output neurons
        # interpolateCorrection 1 action, 1 output neuron (probability)
        self._last_layer = pt.nn.Linear(self._n_neurons, self._n_output)

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        # print("INPUT SHAPE = ", x.size())
        for layer in self._layers:
            x = self._activation(layer(x))

        # map the 1st output to intervall [0, 1] since we want a binary probability (interpolateCorrection)
        bin_choice = pt.sigmoid(self._last_layer(x)[:, 0]).unsqueeze(0)

        # use the remaining output neurons for classification for smoother, all probabilities add up to 1
        # transpose because dim 0 can be different when cat (bin_choice & classification), but here dim 1 is different
        classification = pt.nn.functional.softmax(self._last_layer(x)[:, 1:], dim=1).transpose(0, 1)

        # print("EXPECTED OUTPUT SIZE: ", pt.nn.functional.softmax(self._last_layer(x), dim=1).size(), "\n")

        # the output size of pt.cat([bin_choice,  classification], dim=0) = [7, len_traj],
        # but we want [len_traj, 7], so transpose
        return pt.cat([bin_choice,  classification], dim=0).transpose(0, 1)

    @pt.jit.ignore
    def _scale(self, actions: pt.Tensor) -> pt.Tensor:
        return (actions - self._action_min) / (self._action_max - self._action_min)

    @pt.jit.ignore
    def predict(self, states: pt.Tensor, actions: pt.Tensor) -> pt.Tensor:
        # size(out) = [len_traj, n_probs], size(actions) = [len_traj, n_actions]
        out = self.forward(states)

        # Bernoulli distribution for binary choice ('interpolateCorrection' corresponds to 1st output neuron)
        # unsqueeze() because out[:, 0].size() = (len_trajectory, )
        distr1 = pt.distributions.Bernoulli(out[:, 0].unsqueeze(-1))

        # categorical distribution for classification ('smoother' corresponds to the remaining output neurons)
        # no unsqueeze() because out[:, 1:].size() is already = (len_trajectory, n_smoother)
        distr2 = pt.distributions.Categorical(out[:, 1:])

        # in case of 'interpolateCorrection', we get 1 prob for each point in trajectory
        log_p1 = distr1.log_prob(actions[:, 0].unsqueeze(-1))

        # else or out tensor is already 2D, so we don't need to unsqueeze, otherwise the distr has 1 dim too much,
        # size(out) = [len_traj, n_smoother], size(actions) = [len_traj, 1]
        # TODO: kind weird because behavior should be the same as smoother only... maybe dims are not correct yet
        log_p2 = distr2.log_prob(actions[:, 1]).unsqueeze(-1)

        # merge the log-probs & entropies
        # TODO: not sure if it makes sense, but it was done for fluidic pinball / flow control as well ...
        log_p = pt.cat([log_p1, log_p2], dim=1)
        entropies = pt.cat([distr1.entropy(), distr2.entropy().unsqueeze(-1)], dim=1)

        return log_p.sum(dim=1), entropies.sum(dim=1)


class FCValue(pt.nn.Module):
    def __init__(self, n_states: int, n_layers: int = 2, n_neurons: int = 64,
                 activation: Callable = pt.nn.functional.relu):
        super(FCValue, self).__init__()
        self._n_states = n_states
        self._n_layers = n_layers
        self._n_neurons = n_neurons
        self._activation = activation

        # set up value network
        self._layers = pt.nn.ModuleList()
        self._layers.append(pt.nn.Linear(self._n_states, self._n_neurons))
        if self._n_layers > 1:
            for hidden in range(self._n_layers - 1):
                self._layers.append(pt.nn.Linear(
                    self._n_neurons, self._n_neurons))
        self._layers.append(pt.nn.Linear(self._n_neurons, 1))

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        for i_layer in range(len(self._layers) - 1):
            x = self._activation(self._layers[i_layer](x))
        return self._layers[-1](x).squeeze()


class Agent(ABC):
    """Common interface for all agents.
    """

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save_state(self):
        pass

    @abstractmethod
    def load_state(self):
        pass

    @abstractmethod
    def trace_policy(self):
        pass

    @abstractproperty
    def history(self):
        pass

    @abstractproperty
    def state(self):
        pass
