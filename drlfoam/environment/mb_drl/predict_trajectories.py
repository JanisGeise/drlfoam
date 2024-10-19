"""
class for executing the prediction of trajectories using environment models
"""
import sys
import logging
import argparse
import torch as pt

from typing import List
from os.path import join
from os import chdir, environ

BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)

from ...constants import DEFAULT_TENSOR_TYPE

logger = logging.getLogger(__name__)
pt.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


class PredictTrajectories:
    def __init__(self, train_path: str, n_t_input: int = None, len_trajectory: int = None, n_models: int = None):
        """
        class for predicting trajectories using environment models

        :param train_path: path to the training directory
        :param n_t_input: number of initial time steps used as starting point to predict the trajectories
        :param len_trajectory: trajectory length
        :param n_models: number of environment models
        """
        self._path = train_path
        self._n_t_input = n_t_input
        self._len_trajectory = len_trajectory
        self._shape = (self._len_trajectory, n_models)

        self._dev = "cuda" if pt.cuda.is_available() else "cpu"
        self._batch_size = 2

    def predict(self, env_model: list, initial_states: dict, min_max_values: dict, episode: int, traj_no: int,
                n_states: int, n_actions: int, model_no: int = None) -> dict:
        """
        predict a trajectory based on given starting point

        :param env_model: model ensemble containing the environment models
        :param initial_states: initial states used as starting point
        :param min_max_values: min. and max. values used for normalization within the dataloader class
        :param episode: current episode
        :param traj_no: trajectory number to take from the initial states dict
        :param n_states: number of states
        :param n_actions: number of actions
        :param model_no: number of environment model to use if trajectory should be generated with specific model;
                         if 'None' then for each new time step a model is chosen randomly out of the ensemble
        :return: dict containing the trajectory
        """
        # loop over initial states and replace with the chosen trajectory no
        initial_states = {k: initial_states[k][:, traj_no, :] for k in initial_states.keys()}

        # for each model of the ensemble: load the current state dict
        for model in range(len(env_model)):
            env_model[model].load_state_dict(pt.load(join(self._path, "env_model", f"bestModel_no{model}_val.pt")))

        # load current policy network (saved at the end of the previous episode)
        policy_model = (pt.jit.load(open(join(self._path, f"policy_trace_{episode - 1}.pt"), "rb"))).eval()

        # allocate empty tensors for storing the generated trajectories; since batch normalization only works for
        # batch size > 1 we need to generate two (identical) trajectories
        _predictions = pt.zeros((self._batch_size, self._len_trajectory,
                                 env_model[0].n_outputs + n_actions)).to(self._dev)

        # fill in the initial states TODO: this needs to be generalized once other model types are implemented
        _alpha = pt.zeros((self._batch_size, self._len_trajectory, n_actions))
        _beta = pt.zeros((self._batch_size, self._len_trajectory, n_actions))
        for i in range(self._batch_size):
            _predictions[i, :self._n_t_input, :] = pt.cat([initial_states["states"], initial_states["cy"],
                                                           initial_states["cx"], initial_states["actions"]],
                                                          dim=1).to(self._dev)
            _alpha[i, :self._n_t_input, :] = initial_states["alpha"].to(self._dev)
            _beta[i, :self._n_t_input, :] = initial_states["beta"].to(self._dev)

        # loop over the trajectory, each iteration shift the input window by one time step
        for t in range(self._len_trajectory - self._n_t_input):
            if model_no is None:
                # randomly choose an environment model to make a prediction if no model is specified
                tmp_env_model = env_model[pt.randint(low=0, high=len(env_model), size=(1, 1)).item()]
            else:
                tmp_env_model = env_model[model_no]

            # make prediction and add to predictions
            _pred_states = tmp_env_model(_predictions[:, t:t + self._n_t_input, :].flatten(1)).squeeze().detach()
            _predictions[:, t + self._n_t_input, :-n_actions] = _pred_states

            # use predicted (new) state to get an action for both environment models as new input
            # note: policy network uses real states as input (not scaled to [0, 1]), policy training currently on cpu
            _states_unscaled = self._rescale(_predictions[:, t + self._n_t_input, :n_states], min_max_values["states"])
            _pred_action = policy_model(_states_unscaled.to("cpu")).squeeze().detach()

            # sample the value for omega (are already in [0, 1], so we don't need to rescale them)
            _alpha[:, t + self._n_t_input, :] = _pred_action[:, :n_actions]
            _beta[:, t + self._n_t_input, :] = _pred_action[:, n_actions:]
            beta_distr = pt.distributions.beta.Beta(_alpha[:, t + self._n_t_input, :], _beta[:, t + self._n_t_input, :])

            # add the actions
            _predictions[:, t + self._n_t_input, -n_actions:] = beta_distr.sample()

        # reshape to actions, states, etc. We only need the first trajectory since they are identical
        # further, n_actions = n_cy = n_cx
        trajectories = {"states": _predictions[0, :, :n_states], "actions": _predictions[0, :, -n_actions:],
                        "cy": _predictions[0, :, n_states:n_states+n_actions],
                        "cx": _predictions[0, :, n_states+n_actions:n_states+2*n_actions],
                        "alpha": _alpha[0, :, :].to("cpu"), "beta": _beta[0, :, :].to("cpu")}

        # re-scale everything for PPO training and sort into dict
        for k in trajectories.keys():
            if k != "alpha" and k != "beta":
                trajectories[k] = self._rescale(trajectories[k], min_max_values[k]).to("cpu")

        return trajectories

    def predict_for_each_model(self, model_ensemble: list, initial_states: dict, min_max: dict, n_actions: int,
                               n_states: int, e: int, no: int) -> List[dict]:
        """
        predict the same trajectory but under the usage of only a single model. Then loop over all models to quantify
        the uncertainty between the model predictions to assess if we need to switch back to CFD

        :param model_ensemble: model ensemble containing the environment models
        :param initial_states: initial states used as starting point
        :param min_max: min. and max. values used for normalization within the dataloader class
        :param n_states: number of states
        :param n_actions: number of actions
        :param e: current episode
        :param no: trajectory number to take from the initial states dict
        :return: predicted states and actions for each model when using a single model to predict
        """
        # loop over all models and execute the prediction for ach model
        pred = []
        for model in range(len(model_ensemble)):
            pred.append(self.predict(model_ensemble, initial_states, min_max, e, no, n_states, n_actions,
                                     model_no=model))

        return pred

    def predict_slurm(self, no: int, pred_id: int) -> None:
        """
        execute the prediction for a single trajectory when the executer is SLURM

        :param no: trajectory number to take from the initial states dict
        :param pred_id: file id to load and save the correct data, also used as seed value
        :return: None
        """
        settings = pt.load(join(self._path, "settings_prediction.pt"))
        initial_states = pt.load(join(self._path, "initial_states.pt"))

        # set the properties
        self._len_trajectory = settings["len_traj"]
        self._n_t_input = settings["n_input"]
        self._shape = (self._len_trajectory, settings["n_models"])

        # ensure reproducibility ('no' is chosen in predict_trajectories.py)
        pt.manual_seed(no)
        if pt.cuda.is_available():
            pt.cuda.manual_seed_all(no)

        # predict trajectory
        pred = self.predict(settings["env_model"], initial_states, settings["min_max"], settings["episode"], no,
                            settings["n_states"], settings["n_actions"])

        # execute prediction for each model to assess model performance
        pred_model = self.predict_for_each_model(settings["env_model"], initial_states, settings["min_max"],
                                                 settings["n_actions"], settings["n_states"], settings["episode"], no)

        # save the trajectory, status and uncertainty wrt model the rewards are computed in execute_prediction
        pt.save({"pred": pred, "pred_model": pred_model}, join(self._path, f"prediction_no{pred_id}.pt"))

    @staticmethod
    def _rescale(x: pt.Tensor, min_max_x: list) -> pt.Tensor:
        """
        reverse the normalization of the data

        :param x: normalized data
        :param min_max_x: min- and max-value used for normalizing the data
        :return: rescale data as tensor
        """
        # x = (x_max - x_min) * x_norm + x_min
        return (min_max_x[1] - min_max_x[0]) * x + min_max_x[0]


if __name__ == "__main__":
    ag = argparse.ArgumentParser()
    ag.add_argument("-i", "--id", required=True, help="process id")
    ag.add_argument("-n", "--number", required=True, help="number of the trajectory containing the initial states")
    ag.add_argument("-p", "--path", required=True, type=str, help="path to training directory")
    args = ag.parse_args()

    # instantiate class, cwd = 'drlfoam/drlfoam/environment/mb_drl', so go back to the examples directory
    chdir(join(BASE_PATH, "examples"))
    executer_prediction_slurm = PredictTrajectories(str(join(BASE_PATH, "examples", args.path)))
    executer_prediction_slurm.predict_slurm(int(args.number), int(args.id))
