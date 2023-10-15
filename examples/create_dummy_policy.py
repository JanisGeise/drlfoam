"""Create a randomly initialized policy network.
"""
import sys
from os import environ
from os.path import join

BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)
import torch as pt
from drlfoam.agent import FCPolicy

# 'smoother' only: action_max: 6 smoother available (0, ..., 5), but 1 action (selection of smoother is only parameter)
# policy = FCPolicy(n_states=7, n_actions=1, action_min=pt.zeros(1,), action_max=pt.ones(1,)*5, n_output=6)

# 'interpolateCorrection' only
# policy = FCPolicy(n_states=7, n_actions=1, action_min=pt.zeros(1,), action_max=pt.ones(1,), n_output=1)

# combination of 'interpolateCorrection' and 'smoother', n_actions not used anymore, everything is defined with n_output
# policy = FCPolicy(n_states=6, n_actions=2, action_min=pt.zeros(1,), action_max=pt.ones(1,)*6, n_output=7)

# combination of 'interpolateCorrection', 'smoother' & 'nCellsInCoarsestLevel'
policy = FCPolicy(n_states=6, n_actions=3, action_min=pt.zeros(1,), action_max=pt.ones(1,)*32, n_output=32)

# test policy output -> choose randomly 6 features as iinput
# prediction_test = policy.forward(pt.rand((1, 6)))

script = pt.jit.script(policy)
script.save(join("..", "openfoam", "test_cases", "cylinder2D", "policy.pt"))
script.save(join("..", "openfoam", "test_cases", "weirOverflow", "policy.pt"))
