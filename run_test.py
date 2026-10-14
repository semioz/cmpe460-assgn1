# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! Do not change this file. This is a sample file that
# !    will be run during evaluation of your submission. It
# !    is shared with you so that you can test your implementation
# !    locally. If you get any errors when running this file with
# !    `python run_test.py --valid`, it is highly likely that you
# !    will receive errors during evaluation.
# !
# ! Note that recieving errors will result in a zero score for this
# !    assignment. Therefore, make sure that your implementation is
# !    correct by testing this file locally.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

from data_utils import Dataset, Preprocessor
from sklearn.metrics import f1_score
from model import Model
import pandas as pd
import numpy as np
import argparse
import torch
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running experiments for the pretrained model")
    parser.add_argument("--valid", action="store_true", help="Evaluate validation data")
    parser.add_argument("--test",  action="store_true", help="Evaluate test data")
    args = parser.parse_args()

    if args.test and not os.path.exists("data/test.csv"):
        raise FileNotFoundError("Test data not found. Try it with --valid flag")

    if not os.path.exists("saved_objects/model.pt"):
        raise FileNotFoundError("Either model is not found or named wrong. Save your model object first and name it model.pt")

    features = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
                'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
                'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1',
                'ShapeFactor2', 'ShapeFactor3','ShapeFactor4']
    preprocessor = Preprocessor(feature_columns=features, label_column="Class")

    if args.valid:
        dataset = Dataset("./data/validation.csv", preprocessor, batch_size=32, shuffle=False)
        dataset = dataset.prepare_data()

    elif args.test:
        dataset = Dataset("./data/test.csv", preprocessor, batch_size=32, shuffle=False)
        dataset = dataset.prepare_data()


    # if you use different input dimensionality, mention it in the README.md file
    model = Model(input_dim=16, output_dim=7, device="cuda", pretrained_path="./saved_objects/model.pt")

    performance = 0.0
    with torch.no_grad():
        for X, y in dataset:
            X, y = X.to("cuda"), y.to("cuda")
            prediction = model(X)
            performance += f1_score(np.argmax(y.cpu().detach().numpy(), axis=1),
                                    np.argmax(prediction.cpu().detach().numpy(), axis=1),
                                    average="macro")
    performance = performance / len(dataset)
    print("Evaluation result: ", performance)
