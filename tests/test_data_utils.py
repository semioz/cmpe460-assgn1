import unittest

import numpy as np
import pandas as pd

from data_utils import Dataset, Preprocessor


FEATURES = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
    'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1',
    'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4',
]
LABELS = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']


class PreprocessorTests(unittest.TestCase):
    def test_training_statistics_and_types(self):
        preprocessor = Preprocessor(FEATURES, 'Class')
        features, labels = preprocessor.apply_preprocess(pd.read_csv('data/train.csv'))
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(labels.dtype, np.float32)
        self.assertEqual(features.shape, (8710, 16))
        self.assertEqual(labels.shape, (8710, 7))
        np.testing.assert_allclose(features.mean(axis=0), 0, atol=1e-4)
        np.testing.assert_allclose(features.std(axis=0), 1, atol=1e-4)
        np.testing.assert_array_equal(labels.sum(axis=1), 1)

    def test_fresh_validation_preprocessor_uses_same_scaling_and_classes(self):
        data = pd.read_csv('data/validation.csv')
        preprocessor = Preprocessor(FEATURES, 'Class')
        features, labels = preprocessor.apply_preprocess(data)
        self.assertEqual(np.argmax(labels[0]), LABELS.index(data['Class'][0]))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(abs(features[:, 0].mean()), 0.01)
        batch_features, batch_labels = next(iter(Dataset('data/validation.csv', preprocessor, 32, False).prepare_data()))
        self.assertEqual(tuple(batch_features.shape), (32, 16))
        self.assertEqual(tuple(batch_labels.shape), (32, 7))

    def test_nonfinite_feature_is_rejected(self):
        data = pd.read_csv('data/train.csv', nrows=1)
        data.loc[0, 'Area'] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            Preprocessor(FEATURES, 'Class').apply_preprocess(data)

    def test_unknown_class_is_rejected(self):
        data = pd.read_csv('data/train.csv', nrows=1)
        data.loc[0, 'Class'] = 'UNKNOWN'
        with self.assertRaisesRegex(ValueError, 'UNKNOWN'):
            Preprocessor(FEATURES, 'Class').apply_preprocess(data)


if __name__ == '__main__':
    unittest.main()
