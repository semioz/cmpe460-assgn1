# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! This is the file for the trainer object that would be used for
# !     training the given model using the given optimizer and loss
# !     function.
# !
# !     You need to implement the train_loop and eval_loop functions.
# !     You can also use additional packages in this file.
# !     If necessary, you can also add another functions as required.
# !
# ! Make sure that the final implementation is compatible with the
# !     other classes.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

from torch.utils.data import DataLoader
import torch

from typing import Dict
import wandb

class Trainer:
    def __init__(self, model, optimizer, loss_fn, project_name, device, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.project_name = project_name
        self.device = device

        self.__dict__.update(kwargs)
        self.model = self.model.to(self.device)

    # You can update the following train function to apply regulation during the training process
    def train(self, training_data: DataLoader, validation_data: DataLoader, epochs: int):
        with wandb.init(project=self.project_name) as run:
            for epoch in range(epochs):
                train_result = self.train_loop(training_data)
                eval_result  = self.eval_loop(validation_data)

                self.log_wandb(run, epoch, train_result, eval_result)

    ######################################################################################
    # ! Update Trainer functions (train_loop, eval_loop and log_wandb)
    ######################################################################################
    # Make sure that your output of this function is a dictionary presenting the results of
    # performance metrics for single training loop
    def train_loop(self, data: DataLoader) -> Dict:
        raise NotImplementedError("!! Implementation for Trainer.train_loop is missing...")


    # Make sure that your output of this function is a dictionary presenting the results of
    # performance metrics for single evaluation loop
    def eval_loop(self, data: DataLoader) -> Dict:
        raise NotImplementedError("!! Implementation for Trainer.eval_loop is missing...")


    def log_wandb(self, run, epoch, training_result, eval_result):
        raise NotImplementedError("!! Implementation for Traininer.log_wandb is missing...")
