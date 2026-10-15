# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! This is the file for the model class.
# !     You need to implement the init_model and forward pass functions.
# !     You can also use additional packages in this file.
# !     If necessary, you can also add another functions as required.
# !
# ! Make sure that the final implementation is compatible with the
# !     other classes. Be careful about the input and output types of
# !     the methods.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

from typing import List, Dict

import numpy as np
import torch

class Model(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, pretrained_path: str = None, device: str = "cpu", **kwargs):
        super(Model, self).__init__()

        self.pretrained_path = pretrained_path
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.__dict__.update(kwargs)

        # Initialize self.model object
        self.init_model()

        # Load model if pretrained model is available
        if pretrained_path != None:
            # load cuda-saved weights on the requested device, including cpu.
            self.load_state_dict(torch.load(self.pretrained_path, map_location=self.device, weights_only=True))

        self.to(self.device)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    ##########################################################################
    # ! Update Model functions (init_model and forward)
    ##########################################################################
    def init_model(self):
        self.register_buffer('input_mean', torch.zeros(self.input_dim))
        self.register_buffer('input_std', torch.ones(self.input_dim))
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, self.output_dim),
        )

    def forward(self, x):
        return self.model((x - self.input_mean) / self.input_std)
