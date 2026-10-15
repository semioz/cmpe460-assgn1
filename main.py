# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! This is a sample file for your experiments. You can change
# !     this file as you wish.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
from data_utils import Preprocessor, Dataset
from trainer_utils import Trainer
from model import Model

import os
import torch

features = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
            'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
            'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1',
            'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']


def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    preprocessor = Preprocessor(feature_columns=features, label_column='Class')
    validation_data = Dataset('./data/validation.csv', preprocessor, 32, False).prepare_data()
    # fit normalization on training data only; the saved model reuses it at evaluation.
    training_data = Dataset('./data/train.csv', preprocessor, 32, False).prepare_data()
    train_features = training_data.dataset.tensors[0]
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    os.makedirs('saved_objects', exist_ok=True)

    # we need to kkeep the best validation checkpoint across all experiments.
    best_f1 = 0.0
    experiments = [
        {'learning_rate': 1e-3, 'weight_decay': 0, 'batch_size': 32, 'epochs': 30, 'seed': 42},
        {'learning_rate': 5e-4, 'weight_decay': 1e-4, 'batch_size': 64, 'epochs': 30, 'seed': 42},
        {'learning_rate': 2e-3, 'weight_decay': 1e-4, 'batch_size': 32, 'epochs': 30, 'seed': 42},
        {'learning_rate': 1e-3, 'weight_decay': 1e-4, 'batch_size': 32, 'epochs': 50, 'seed': 42},
        {'learning_rate': 2e-3, 'weight_decay': 1e-4, 'batch_size': 32, 'epochs': 50, 'seed': 43},
        {'learning_rate': 5e-4, 'weight_decay': 1e-4, 'batch_size': 64, 'epochs': 60, 'seed': 42},
    ]
    for config in experiments:
        torch.manual_seed(config['seed'])
        training_data = Dataset('./data/train.csv', preprocessor, config['batch_size'], True).prepare_data()
        model = Model(input_dim=16, output_dim=7, device=device)
        with torch.no_grad():
            model.input_mean.copy_(mean.to(device))
            model.input_std.copy_(std.to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        trainer = Trainer(model, optimizer, torch.nn.CrossEntropyLoss(), 'cmpe460-assignment01', device,
                          entity='semioz', config=config)
        best_f1 = trainer.train(training_data, validation_data, epochs=config['epochs'], best_f1=best_f1)
        print(f"{config}: best validation evaluator F1 = {best_f1:.4f}", flush=True)


if __name__ == '__main__':
    main()
