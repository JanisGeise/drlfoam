"""Create a randomly initialized policy network.
"""
import sys
from os import environ
from os.path import join

BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)
import torch as pt
from drlfoam.agent import FCPolicy


policy = FCPolicy(n_states=7, n_actions=1, action_min=pt.zeros(1,), action_max=pt.ones(1,))
script = pt.jit.script(policy)
script.save(join("..", "openfoam", "test_cases", "cylinder2D", "policy.pt"))
