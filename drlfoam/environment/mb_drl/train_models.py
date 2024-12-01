"""
training routine for environment models
"""
import os
import sys
import argparse
import logging
from typing import Tuple

import torch as pt

from os import chdir, mkdir
from os.path import join, exists
from torch.utils.data import DataLoader

BASE_PATH = os.environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)

from ...constants import DEFAULT_TENSOR_TYPE

logger = logging.getLogger(__name__)
pt.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


class TrainModelEnsemble:
    def __init__(self, train_path: str, env: str, max_epochs: int = 2500, lr: float = 0.01,
                 stop_abs_value: float = 1e-6, patience: float = 25, check_every: int = 1):
        """
        implements a class for executing the model training

        :param train_path: path to the directory of the current training
        :param env: environment, either 'local# or 'slurm'
        :param max_epochs: max. number of epochs to run the model training
        :param lr: initial learning rate
        :param stop_abs_value: stop training when validation loss reaches this value
        :param patience: wait N epochs for the validation loss to improve, if not stop the training;
                         gets reset each time a new best validation loss is reached
        :param check_every: check the stopping criteria every N epochs
        """
        self._check_every = check_every
        self._train_path = train_path
        self._env = env
        self._max_epochs = max_epochs
        self._lr = lr
        self._min_lr = 1.0e-4
        self._weight_decay = 1e-3
        self._stop_abs = stop_abs_value
        self._patience = patience
        self._patience_counter = 0

        self._dev = "cuda" if pt.cuda.is_available() else "cpu"
        self._save_dir = join(self._train_path, "env_model")
        self._save_name = "bestModel_no"

        # create directory for model-training, when running on HPC multiple processes try to create this directory at
        # the same time so make sure this doesn't cause any issues
        if not exists(self._save_dir):
            try:
                mkdir(self._save_dir)
            except FileExistsError:
                logger.info(f"Directory '{self._save_dir}' was already created by another process.")

        self._criterion = pt.nn.MSELoss()

    def train_model_ensemble(self, model: pt.nn.Module, training_data: DataLoader, validation_data: DataLoader,
                             load: bool = False, model_no: int = 0) -> list or None:
        """
        wrapper function for handling the training of the model-ensemble

        :param model: environment model
        :param training_data: dataloader containing the training data
        :param validation_data: dataloader containing the validation data
        :param load: flag for loading an existing (pre-trained) model (True)
        :param model_no: number of the current environment model within the ensemble
        :return: list with training and validation loss if executer is local else the losses are saved as .pt file
        """
        # train and validate environment models with CFD data from the previous episode
        logger.info(f"Start training environment model no. {model_no}")

        # load environment models trained in the previous CFD episode
        if load:
            model.load_state_dict(pt.load(join(self._save_dir, f"{self._save_name}{model_no}_val.pt")))

        # train environment models
        if self._env == "local":
            train_loss, val_loss = self._train_model(model.to(self._dev), training_data, validation_data, no=model_no)

            return [train_loss, val_loss]

        else:
            self._train_model(model.to(self._dev), training_data, validation_data, no=model_no)

    def train_model_ensemble_slurm(self, model_no: int) -> None:
        """
        executes the model training on an HPC cluster using SLURM

        :param model_no: number of the current environment model
        :return: None
        """
        # initialize each model with different seed value
        pt.manual_seed(model_no)
        if pt.cuda.is_available():
            pt.cuda.manual_seed_all(model_no)

        # load the settings and dataloader
        settings = pt.load(join(self._train_path, "settings_model_training.pt"))
        loader_train = pt.load(join(self._train_path, "loader_train.pt"))
        loader_val = pt.load(join(self._train_path, "loader_val.pt"))

        # execute the training
        self.train_model_ensemble(settings["env_model"], loader_train[model_no], loader_val[model_no],
                                  load=settings["load"], model_no=model_no)

    def _train_model(self, model: pt.nn.Module, dataloader_train: DataLoader, dataloader_val: DataLoader,
                     no: int) -> Tuple[list, list] or None:
        """
        train a single environment model

        :param model: environment model
        :param dataloader_train: dataloader containing the training data
        :param dataloader_val: dataloader containing the validation data
        :param no: number of the current environment model within the ensemble
        :return: training and validation loss if executed local, else the losses are saved in the training directory
        """
        # optimizer settings
        optimizer = pt.optim.AdamW(params=model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, min_lr=self._min_lr)

        # lists for storing losses
        best_val_loss, best_train_loss = 1.0e5, 1.0e5
        training_loss, validation_loss = [], []
        self._patience_counter = 0

        for epoch in range(1, self._max_epochs + 1):
            t_loss_tmp, v_loss_tmp = [], []

            # training loop
            model.train()
            for feature, label in dataloader_train:
                optimizer.zero_grad()
                prediction = model(feature).squeeze()
                loss_train = self._criterion(prediction, label.squeeze())
                loss_train.backward()
                optimizer.step()
                t_loss_tmp.append(loss_train.item())
            training_loss.append(pt.mean(pt.tensor(t_loss_tmp)).to("cpu").detach())

            # validation loop
            model.eval()
            with pt.no_grad():
                for feature, label in dataloader_val:
                    prediction = model(feature).squeeze()
                    loss_val = self._criterion(prediction, label.squeeze())
                    v_loss_tmp.append(pt.mean(loss_val).item())
            validation_loss.append(pt.mean(pt.tensor(v_loss_tmp)).to("cpu"))

            scheduler.step(metrics=validation_loss[-1])

            if validation_loss[-1] < best_val_loss:
                pt.save(model.state_dict(), join(self._save_dir, self._save_name + f"{no}_val.pt"))
                best_val_loss = validation_loss[-1]
                self._patience_counter = 0
            else:
                self._patience_counter += 1

            # print some info every 100 epochs
            if epoch % 100 == 0:
                logger.info(f"epoch {epoch}, avg. train loss = " +
                            "{:8f}".format(pt.mean(pt.tensor(training_loss[-self._check_every:])).item()) +
                            f", avg. validation loss = " +
                            "{:8f}".format(pt.mean(pt.tensor(validation_loss[-self._check_every:])).item()))

            # check every N epochs if model performs well on validation data or validation loss converges
            if epoch % self._check_every == 0:
                if validation_loss[-1] <= self._stop_abs or self._patience_counter >= self._patience:
                    logger.info(f"Stopping training after {epoch} epochs due to early stopping criteria.")
                    break

        if self._env == "local":
            return training_loss, validation_loss
        else:
            pt.save(training_loss, join(self._save_dir, f"loss{no}_train.pt"))
            pt.save(validation_loss, join(self._save_dir, f"loss{no}_val.pt"))


if __name__ == "__main__":
    ag = argparse.ArgumentParser()
    ag.add_argument("-m", "--model", required=True, help="number of the current environment model")
    ag.add_argument("-p", "--path", required=True, type=str, help="path to training directory")
    args = ag.parse_args()

    # # instantiate class, cwd = 'drlfoam/drlfoam/environment/mb_drl', so go back to the examples directory
    chdir(join(BASE_PATH, "examples"))
    executer_training_slurm = TrainModelEnsemble(str(join(BASE_PATH, "examples", args.path)), "slurm")
    executer_training_slurm.train_model_ensemble_slurm(int(args.model))
