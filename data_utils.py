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
    # Training-set mean and population standard deviation; fixed for validation and test.
    FEATURE_STATS = {
        'Area': (52992.9001148, 29382.718634),
        'Perimeter': (854.415982778, 214.814115276),
        'MajorAxisLength': (319.752544836, 85.8317090184),
        'MinorAxisLength': (202.19348921, 45.1686746697),
        'AspectRation': (1.58176571463, 0.245367517971),
        'Eccentricity': (0.750564210828, 0.0917729120933),
        'ConvexArea': (53708.1843858, 29820.016134),
        'EquivDiameter': (252.874034615, 59.3917387703),
        'Extent': (0.749538450992, 0.0487143590125),
        'Solidity': (0.987174165477, 0.00462893637659),
        'roundness': (0.873734345218, 0.0595256139582),
        'Compactness': (0.800191424835, 0.0614567431449),
        'ShapeFactor1': (0.00656822913895, 0.00113437069712),
        'ShapeFactor2': (0.00171966628705, 0.000595923539568),
        'ShapeFactor3': (0.644083247657, 0.0986241128227),
        'ShapeFactor4': (0.995107557259, 0.00433933766242),
    }
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
        features = np.asarray(data, dtype=np.float64)
        if not np.isfinite(features).all():
            raise ValueError('Features must be finite')
        mean, std = np.array([self.FEATURE_STATS[name] for name in self.feature_columns]).T
        return ((features - mean) / std).astype(np.float32)

    def prepare_labels(self, data: pd.DataFrame) -> np.ndarray:
        labels = data[self.label_column].map({name: i for i, name in enumerate(self.LABELS)})
        if labels.isna().any():
            unknown = data.loc[labels.isna(), self.label_column].unique().tolist()
            raise ValueError(f'Unknown class labels: {unknown}')
        return np.eye(len(self.LABELS), dtype=np.float32)[labels.to_numpy(dtype=np.int64)]
