# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! This is a sample file for your experiments. You can change
# !     this file as you wish.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
from data_utils import Preprocessor, Dataset
from trainer_utils import Trainer
from model import Model

import torch

features = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
            'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
            'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1',
            'ShapeFactor2', 'ShapeFactor3','ShapeFactor4']
preprocessor = Preprocessor(feature_columns=features, label_column="Class")

training_dataset = Dataset("./data/train.csv", preprocessor, batch_size=32, shuffle=True)
training_dataset = training_dataset.prepare_data()

validation_dataset = Dataset("./data/validation.csv", preprocessor, batch_size=32, shuffle=False)
validation_dataset = validation_dataset.prepare_data()

test_dataset = Dataset("./data/test.csv", preprocessor, batch_size=32, shuffle=False)
test_dataset = test_dataset.prepare_data()


model = Model(input_dim=16, output_dim=7, device="cuda", pretrained_path=None)
loss_fn   = None
optimizer = None

trainer = Trainer(model, optimizer, loss_fn, "test_model", device="cuda")
trainer.train(training_dataset, validation_dataset, epochs=100)

model.save("./saved_objects/model.pt")
