# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! This is the file for dataset and preprocessing objects.
# !
# !     Do not make any changes in Dataset class!
# !     You need to update Preprocessor classes.
# !
# !     You can add another class or method as needed.
# !     You can also add required parameters to the methods.
# !     You can also implement additional methods as required.
# !     You can also use additional packages in this file.
# !
# ! Make sure that the final implementation is compatible with the
# !     other classes and its methods. Be careful about the
# !     input and output types of the methods.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

from torch.utils.data import TensorDataset, DataLoader
import torch

from typing import List, Tuple
import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, path, preprocessor, batch_size, shuffle):
        self.path = path
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.shuffle = shuffle

    def load_data(self):
        data = pd.read_csv(self.path)
        return data

    def prepare_data(self):
        data = self.load_data()
        X, y = self.preprocessor.apply_preprocess(data)

        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
        dataset = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataset


class Preprocessor:
    LABELS = ('BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA')

    def __init__(self, feature_columns: List, label_column: str, **kwargs):
        self.feature_columns = feature_columns
        self.label_column = label_column

        self.__dict__.update(kwargs)

    def apply_preprocess(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = self.select_features(data)
        X = self.preprocess_features(X)

        y = self.prepare_labels(data)

        return X, y

    def select_features(self, data: pd.DataFrame) -> np.ndarray:
        return data[self.feature_columns].values

    ##########################################################################
    # ! Update Preprocessor functions (preprocess_feature and select_labels)
    ##########################################################################
    def preprocess_features(self, data: np.ndarray) -> np.ndarray:
        features = np.asarray(data, dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError('Features must be finite')
        return features

    def prepare_labels(self, data: pd.DataFrame) -> np.ndarray:
        labels = data[self.label_column].map({name: i for i, name in enumerate(self.LABELS)})
        if labels.isna().any():
            unknown = data.loc[labels.isna(), self.label_column].unique().tolist()
            raise ValueError(f'Unknown class labels: {unknown}')
        return np.eye(len(self.LABELS), dtype=np.float32)[labels.to_numpy(dtype=np.int64)]
