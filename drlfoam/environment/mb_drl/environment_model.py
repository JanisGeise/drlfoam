"""
implements a class for emulating the simulation environment
"""
import sys
import logging
import torch as pt

from os import environ
from os.path import join
from typing import Union, List, Tuple
from torch.utils.data import random_split, TensorDataset, DataLoader

from ...agent import PPOAgent
from ...execution import SlurmBuffer
from ...constants import DEFAULT_TENSOR_TYPE

from .fcnn_model import FCNNModel
from .dataloader import TrajectoryLoader
from .compute_statistics import ComputeStatistics
from .execute_prediction import ExecuteModelPrediction
from .execute_model_training import ExecuteModelTraining

logger = logging.getLogger(__name__)
pt.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


class EnvironmentModel:
    def __init__(self, train_path: str, model_type: str, simulation, n_runner_train: int, buffer_size: int, executer: str,
                 agent: PPOAgent, n_models_thr: int = 3, n_models: int = 5, n_input_time_steps: int = 30,
                 slurm_config: SlurmBuffer = None, seed: int = 0, n_runner_pred: int = None):
        """
        implements a wrapper class for handling the model-based part of the PPO training

        :param train_path: path to the training directory
        :param model_type: type of the NN to use; calling 'print_available_models()' displays available options
        :param simulation: environment class from drlfoam.environment, e.g., RotatingCylinder2D or RotatingPinball2D
        :param n_runner_train: number of runners for executing the model training
        :param buffer_size: buffer size
        :param executer: either 'local' or 'slurm'
        :param agent: PPO agent class
        :param n_models_thr: number of models which have to improve the policy; used as criteria to switch back to CFD
        :param n_models: number of models in the ensemble
        :param n_input_time_steps: number of initial time steps used as starting point to predict the trajectories
        :param slurm_config: SLURM config if the training is executed on an HPC cluster
        :param n_runner_train: number of runners for executing the predictions, if None the same as for training is used
        """
        # settings
        self.train_path = train_path
        self._simulation = simulation
        self._n_runner_train = n_runner_train
        self._n_runner_prediction = self._n_runner_train if n_runner_pred is None else n_runner_pred
        self._buffer_size = buffer_size
        self._executer = executer.lower()
        self._agent = agent
        self._n_actions = self._simulation.n_actions
        self._n_states = self._simulation.n_states
        self._trajectory_length = int((simulation.end_time - simulation.start_time) / simulation.control_interval)
        self._slurm_config = None if self._executer == "local" else slurm_config

        # settings for the environment model
        self._n_models = n_models
        self._n_t_input = n_input_time_steps
        self._device = "cuda" if pt.cuda.is_available() else "cpu"
        self._current_episode = 1
        self._models_available = ["FCNN"]
        self._env_model = self._initialize_model(model_type)
        self._split_ratio = 0.75
        self._batch_size = 25
        self._model_ensemble = None

        # ensure reproducibility
        pt.manual_seed(seed)
        if pt.cuda.is_available():
            pt.cuda.manual_seed_all(seed)

        # ensure that there are no round-off errors, which lead to switching
        self._threshold = round(n_models_thr / n_models, 6)

        # class to time everything
        self.statistics = ComputeStatistics()

        # everything regarding CFD data handling
        self._last_cfd = 0
        self._obs_cfd = []
        self._cfd_data = None
        self._initial_states = {}
        self._min_max_values = {}
        self._dataloader = TrajectoryLoader(self._n_states, self._n_actions, self._buffer_size, len(self._obs_cfd),
                                            self._trajectory_length)

        # executer classes for training and prediction
        self._executer_training = ExecuteModelTraining(self.train_path, self._executer, self._env_model, self._n_models,
                                                       self._slurm_config, self._n_runner_train)
        self._executer_prediction = ExecuteModelPrediction(self.train_path, self._executer, self._n_models, self._agent,
                                                           self._buffer_size, self._trajectory_length, self._n_t_input,
                                                           self._simulation, self._slurm_config,
                                                           self._n_runner_prediction, self._dataloader.keys, seed)
        self._policy_loss = []
        self._buffer = []
        self._losses = None
        self.prediction_failed = False

    def print_available_models(self) -> None:
        """
        displays all available model types
        :return: None
        """
        print("Available environment model types: ", self._models_available)

    def train_models(self, e: int) -> None:
        """
        wrapper method for handling everything regarding the model training

        :param e: current episode
        :return: None
        """
        # set the current episode and add it to the list with CFD episodes
        self._current_episode = e
        self._append_cfd_obs()

        # load CFD data from the previous N episodes
        self._cfd_data = self._dataloader.load(self._obs_cfd)

        # extract initial data as starting point for MB episodes
        self._create_initial_data()

        # create feature-label pairs for model training
        self._prepare_data()

        # set up the model training of the current episode and train the models
        self.statistics.start_timer()
        self._losses, self._model_ensemble = self._executer_training.execute()
        self.statistics.time_model_training()
        self._save_losses()

        # reset properties
        self._reset()

        # update the boundaries in which a trajectory generated by the environment models is valid
        self._executer_prediction.check_mb_trajectories.update_last_bounds(self._cfd_data, self._min_max_values)

    def predict(self, e: int) -> None:
        """
        wrapper method for handling everything regarding the trajectories predicted by the environment models

        :param e: current episode
        :return: None
        """
        # check if 'train_models' was already executed, otherwise we don't have models or initial states
        assert self._model_ensemble is not None, ("Couldn't find any model-ensemble to predict trajectories. "
                                                  "The model training must be executed before prediction.")

        # execute the prediction of trajectories and save the policy loss
        self.statistics.start_timer()
        _policy_loss, self._buffer = self._executer_prediction.execute(e, self._model_ensemble,
                                                                       self._initial_states, self._min_max_values)
        self._policy_loss.append(_policy_loss)

        # set a convergence flag in case all predictions were invalid
        if not self._buffer:
            self.prediction_failed = True
        if len(self._buffer) < self._buffer_size:
            logger.info(f"Found only {len(self._buffer)} / {self._buffer_size} converged trajectories.")

        # if we have at least one converged trajectory, update the validity boundaries and decompose the parameters
        # (from, e.g., cx to cx_a), otherwise we switch back to CFD
        else:
            self._executer_prediction.check_mb_trajectories.update_last_bounds(self._buffer, self._min_max_values)

            # save the observations
            self._save_observations()
        self.statistics.time_mb_episode()

    def determine_switching(self, e: int) -> bool:
        """
        check if the environment models are still improving the policy or if it should be switched back to CFD
        to update the environment models

        :param e: current episode
        :return: bool if training should be switched back to CFD to update the environment models
        """
        # update the current episode
        self._current_episode = e

        # we can't use the criteria if we only have a single model, in that case switch every four episodes
        if self._n_models == 1:
            return self._current_episode % 4 == 0

        # we need to have two MB episodes to compute a difference, so don't switch after a single MB episode
        if len(self._policy_loss) < 2:
            switch = 0
        else:
            # check the difference of the policy loss for each mode
            diff = ((pt.tensor(self._policy_loss[-2]) - pt.tensor(self._policy_loss[-1])) > 0.0).int()

            # if the policy for less than x% of the models improves, switch to CFD (0 = no switching, 1 = switch)
            switch = [0 if sum(diff) / len(diff) >= self._threshold else 1][0]

        return True if switch else False

    def _append_cfd_obs(self) -> None:
        """
        append the current path to the observation file to the list of CFD episodes

        :return: None
        """
        self._obs_cfd.append(join(self.train_path, f"observations_{self._current_episode}.pt"))

    def _save_losses(self) -> None:
        """
        save the training and validation loss of the environment models

        :return: None
        """
        # save train- and validation losses of the environment models in case the training didn't crash
        try:
            if self._n_models == 1:
                losses = {"train_loss": self._losses[0], "val_loss": self._losses[1]}
            else:
                losses = {"train_loss": [loss[0] for loss in self._losses if loss[0]],
                          "val_loss": [loss[1] for loss in self._losses if loss[1]]}
        except IndexError:
            losses = {"train_loss": [], "val_loss": []}
        pt.save(losses, join(self.train_path, f"env_model_loss_{self._current_episode}.pt"))

    def _save_observations(self, name: str = "observations"):
        pt.save(self._buffer, join(self.train_path, f"{name}_{self._current_episode}.pt"))

    def _reset(self) -> None:
        """
        reset the policy loss and set the current episode as last CFD episode

        :return: None
        """
        self._policy_loss = []
        self._last_cfd = self._current_episode

    @property
    def observations(self) -> Tuple[list, list, list]:
        """
        get the states, actions and rewards from the model buffer

        :return: Tuple with states, actions and rewards
        """
        # if we have trajectories, get the states and actions from the buffer
        if self._buffer:
            states = [traj["states"] for traj in self._buffer]
            actions = [traj["actions"] for traj in self._buffer]
            rewards = [traj["rewards"] for traj in self._buffer]
        else:
            logger.warning("Couldn't find any trajectories generated by environment models, returning empty buffer.")
            states, actions, rewards = [], [], []
        return states, actions, rewards

    def _initialize_model(self, model_type: str) -> Union[FCNNModel]:
        """
        initialize the environment model, it is assumed that N_cy =N_cx = N_actions

        Note: return type will be updated once more model types are available.

        :param model_type: type of the NN to use; calling 'print_available_models()' displays available options
        :return: the chosen environment model
        """
        if model_type not in self._models_available:
            logger.error(f"Unknown model type '{model_type}'. Available options are: {self._models_available}")

        if model_type == "FCNN":
            # input: N_t * (N_states, N_actions, N_cx, N_cy), assuming n_cy = n_cx = n_actions
            return FCNNModel(self._n_t_input * (self._n_states + 3 * self._n_actions),
                             (self._n_states + 2 * self._n_actions))

    def _prepare_data(self) -> None:
        """
        generate feature-label pairs from the data and create Datasets from them

        :return: None
        """
        # create feature-label pairs for all possible input states of given data
        features, labels = self._generate_feature_labels()

        # reset the dataloader to free up some space
        self._dataloader.reset()

        # create dataset
        data = TensorDataset(features.to(self._device), labels.to(self._device))

        # split into training ind validation data
        n_train = int(self._split_ratio * features.size(0))
        n_val = features.size(0) - n_train
        train, val = random_split(data, [n_train, n_val])

        # Train each model on different subset of the data. In case only a single model is used, it is trained on the
        # complete dataset
        self._executer_training.loader_training = self._create_subset_of_data(train)
        self._executer_training.loader_validation = self._create_subset_of_data(val)

    def _generate_feature_labels(self) -> Tuple[pt.Tensor, pt.Tensor]:
        """
        create feature-label pairs of all available trajectories for FCNN models

        TODO: generalize for other model architectures once other model types are implemented

        :return: tensor with features and tensor with corresponding labels, sorted as [batches, N_features (or labels)]
        """
        feature, label, nt = [], [], self._n_t_input
        shape_input = (self._trajectory_length - nt, nt * (self._n_states + 3 * self._n_actions))

        for n in range(self._buffer_size):
            f, l = pt.zeros(shape_input), pt.zeros(shape_input[0], (self._n_states + 2 * self._n_actions))
            for t_idx in range(self._trajectory_length - nt):
                # [n_probes * n_time_steps * states, n_time_steps * cy, n_time_steps * cx, n_time_steps * action]
                s = self._cfd_data["states"][t_idx:t_idx + nt, n, :].squeeze()
                cy_tmp = self._cfd_data["cy"][t_idx:t_idx + nt, n]
                cx_tmp = self._cfd_data["cx"][t_idx:t_idx + nt, n]
                a = self._cfd_data["actions"][t_idx:t_idx + nt, n]
                f[t_idx, :] = pt.cat([s, cy_tmp, cx_tmp, a], dim=1).flatten()
                l[t_idx, :] = pt.cat([self._cfd_data["states"][t_idx + nt, n, :].squeeze(),
                                      self._cfd_data["cy"][t_idx + nt, n],
                                      self._cfd_data["cx"][t_idx + nt, n]], dim=0)
            feature.append(f)
            label.append(l)

        return pt.cat(feature, dim=0), pt.cat(label, dim=0)

    def _create_subset_of_data(self, data: TensorDataset) -> List[DataLoader]:
        """
        creates a subset of the dataset wrt number of models, so that each model can be trained on a different subset
        of the dataset to accelerate the overall training process

        :param data: the dataset consists of features and labels
        :return: list of the dataloaders created for each model
        """
        rest = len(data.indices) % self._n_models
        idx = [int(len(data.indices) / self._n_models) for _ in range(self._n_models)]

        # distribute the remaining idx equally over the models
        for i in range(rest):
            idx[i] += 1

        return [DataLoader(i, self._batch_size, True, drop_last=False) for i in random_split(data, idx)]

    def _create_initial_data(self) -> None:
        """
        create initial states used as a starting point to predict the trajectories using environment models

        :return: None
        """
        for key, value in self._dataloader.data.items():
            if key.startswith("min_max") and not key.endswith("rewards"):
                self._min_max_values[key.split("_")[-1]] = value
            elif not key.endswith("rewards"):
                self._initial_states[key] = value[:self._n_t_input, :, :]


if __name__ == "__main__":
    pass
