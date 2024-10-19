"""
executer for model training
"""
import sys
import logging
import torch as pt
from typing import Tuple

from glob import glob
from os.path import join
from os import chdir, environ, remove, getcwd


BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)

from ...execution import SlurmConfig
from ...constants import DEFAULT_TENSOR_TYPE
from ...execution.manager import TaskManager
from ...execution.slurm import submit_and_wait

from .fcnn_model import FCNNModel
from .train_models import TrainModelEnsemble

logger = logging.getLogger(__name__)
pt.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


class ExecuteModelTraining:
    def __init__(self, train_path: str, executer: str, env_model: FCNNModel, n_models: int, config: SlurmConfig = None,
                 n_runner: int = None):
        """

        :param train_path: path to the training directory
        :param executer: either 'local' or 'slurm'
        :param env_model: initialized environment model
        :param n_models: number of models in the ensemble
        :param config: SLURm config if training is executed on HPC cluster
        :param n_runner: number of runners if training is executed on HPC cluster
        """
        self._train_path = train_path
        self._executer = executer.lower()
        self._model = env_model
        self._n_models = n_models
        self._config = config
        self._manager = TaskManager(n_runners_max=n_runner)
        self._script_name = "execute_model_training.sh"

        self.loader_validation = None
        self.loader_training = None
        self._load = False
        self._losses = []
        self._model_ensemble = []

        # executer for model training, if SLURM the executer is set directly in the training script
        self._train_models = TrainModelEnsemble(self._train_path, self._executer) if config is None else None

    def execute(self) -> Tuple[list, list]:
        """
        wrapper handling the execution of the model training

        :return: losses for training and validation and the trained model ensemble
        """
        # reset the losses and model buffer before appending new ones
        self._model_ensemble, self._losses = [], []

        # execute the model training
        self._execute_training_slurm() if self._config is not None else self._execute_training_local()

        # set a flag that the models are trained, in future episodes we can re-load these models
        self._load = True

        # in case only a single model is used, extract the losses
        return self._losses[0] if self._n_models == 1 else self._losses, self._model_ensemble

    def _execute_training_local(self) -> None:
        """
        executes the model training if the executer is local

        :return: None
        """
        for m in range(self._n_models):
            # initialize each model training with different seed value
            pt.manual_seed(m)
            if pt.cuda.is_available():
                pt.cuda.manual_seed_all(m)

            # (re-) train each model in the ensemble with max. 2500 epochs
            self._losses.append(self._train_models.train_model_ensemble(self._model, self.loader_training[m],
                                                                        self.loader_validation[m], load=self._load,
                                                                        model_no=m))
            self._model_ensemble.append(self._model.eval())

    def _execute_training_slurm(self) -> None:
        """
        executes the model training if the executer is SLURM

        :return: None
        """
        # save all data as tmp files
        pt.save(self.loader_training, join(self._train_path, "loader_train.pt"))
        pt.save(self.loader_validation, join(self._train_path, "loader_val.pt"))
        pt.save({"train_path": self._train_path, "env_model": self._model, "load": self._load},
                join(self._train_path, "settings_model_training.pt"))

        # save the cwd, because we need to chdir for executing the script
        current_cwd = getcwd()

        # overwrite the last command to the correct script (if we append, we modify config for prediction as well)
        self._config.job_name = "model_train"
        self._config.commands[-1] = "python3 train_models.py -m $1 -p $2"
        self._config.write(join(current_cwd, self._train_path, self._script_name))

        # go to training directory and execute the shell script for model training
        chdir(join(current_cwd, self._train_path))
        for m in range(self._n_models):
            self._manager.add(submit_and_wait, [self._script_name, str(m), self._train_path])
        self._manager.run()

        # then go back to the 'drlfoam/examples' directory and continue training
        chdir(current_cwd)

        for m in range(self._n_models):
            # load the losses once the training is done; in case a job gets canceled, there is no loss available
            try:
                # assuming models are always trained in subdirectory "env_model"
                self._losses.append([pt.load(join(BASE_PATH, self._train_path, "env_model", f"loss{m}_train.pt")),
                                     pt.load(join(BASE_PATH, self._train_path, "env_model", f"loss{m}_val.pt"))])
            except FileNotFoundError:
                self._losses.append([[], []])

            self._model_ensemble.append(self._model.eval())

        # remove tmp data and scripts
        self._reset_slurm()

    def _reset_slurm(self) -> None:
        """
        remove all created tmp files when executed with SLURM

        :return: None
        """
        [remove(f) for f in glob(join(self._train_path, "loader_*.pt"))]
        [remove(f) for f in glob(join(self._train_path, "slurm*.out"))]
        remove(join(self._train_path, "settings_model_training.pt"))
        remove(join(self._train_path, self._script_name))


if __name__ == "__main__":
    pass
