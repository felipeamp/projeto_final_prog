#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright Felipe de Albuquerque Mello Pereira 2017


"""
Module containing all tests for the dataset module.
"""


import os
import sys
import unittest

sys.path.insert(0, '../src')
import dataset

class TestDatasetConfigs(unittest.TestCase):
    """
    Tests loading datasets config, either with a missing field or a correct file.
    """
    def test_nonexistent_file(self):
        """
        Config pointing to non-existent file.
        """
        self.assertIsNone(dataset.load_config('.'))

    def test_missing_name(self):
        """
        Config without dataset_name.
        """
        self.assertIsNone(
            dataset.load_config(os.path.join('.', 'data', 'wrong_configs', 'config_1')))

    def test_missing_key_index(self):
        """
        Config without key_attrib_index.
        """
        self.assertIsNone(
            dataset.load_config(os.path.join('.', 'data', 'wrong_configs', 'config_2')))

    def test_missing_class_index(self):
        """
        Config without class_attrib_index.
        """
        self.assertIsNone(
            dataset.load_config(os.path.join('.', 'data', 'wrong_configs', 'config_3')))

    def test_missing_split_char(self):
        """
        Config without split_char.
        """
        self.assertIsNone(
            dataset.load_config(os.path.join('.', 'data', 'wrong_configs', 'config_4')))

    def test_missing_missing_value_str(self):
        """
        Config without missing_value_str.
        """
        self.assertIsNone(
            dataset.load_config(os.path.join('.', 'data', 'wrong_configs', 'config_5')))

    def test_loading_correct_config(self):
        """
        Correct config.
        """
        self.assertIsInstance(dataset.load_config(os.path.join('.', 'data', 'train_dataset1')),
                              dict)


class TestDatasetWithoutValidAttribs(unittest.TestCase):
    """
    Tests loading a dataset without valid attributes.
    """
    def setUp(self):
        """
        Loads dataset config.
        """
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'dataset_without_valid_attributes'))

    def test_no_valid_attrib(self):
        """
        Tests loading a dataset without valid attributes. Should raise ValueError.
        """
        with self.assertRaises(ValueError):
            dataset.Dataset(self.config["filepath"],
                            self.config["key attrib index"],
                            self.config["class attrib index"],
                            self.config["split char"],
                            self.config["missing value string"],
                            load_numeric=True)


class TestTrainDataset1WithoutNumeric(unittest.TestCase):
    """
    Tests loading dataset without its numeric attributes.
    """
    def setUp(self):
        """
        Loads dataset config and Dataset without numeric attributes.
        """
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset1'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=False)

    def test_valid_attribs(self):
        """
        Tests which attributes are considered valid.
        """
        self.assertEqual(self.data.valid_nominal_attribute, [True, False, True, False])
        self.assertEqual(self.data.valid_numeric_attribute, [False] * 4)
        self.assertEqual(self.data.load_numeric, False)


class TestTrainDataset1WithNumeric(unittest.TestCase):
    """
    Tests loading dataset with its numeric attributes.
    """
    def setUp(self):
        """
        Loads dataset config and Dataset without numeric attributes.
        """
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset1'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=True)

    def test_valid_attribs(self):
        """
        Tests which attributes are considered valid.
        """
        self.assertEqual(self.data.valid_nominal_attribute, [True, False, True, False])
        self.assertEqual(self.data.valid_numeric_attribute, [False, True, False, False])
        self.assertEqual(self.data.load_numeric, True)

    def test_num_samples(self):
        """
        Tests number of samples loaded.
        """
        self.assertEqual(self.data.num_samples, 10)

    def test_num_classes(self):
        """
        Tests number of classes seen.
        """
        self.assertEqual(self.data.num_classes, 2)

    def test_samples_classes(self):
        """
        Tests class seen for each sample.
        """
        self.assertEqual(self.data.sample_class, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    def test_key_attrib_index(self):
        """
        Tests the key_attrib_index.
        """
        self.assertIsNone(self.data.key_attrib_index)

    def test_class_attrib_index(self):
        """
        Tests the class_attrib_index.
        """
        self.assertEqual(self.data.class_attrib_index, -1)

    def test_attrib_names(self):
        """
        Tests the attributes' names.
        """
        self.assertEqual(self.data.attrib_names,
                         ['nominal_attrib1', 'numeric_attrib', 'nominal_attrib2', 'class'])

    def test_class_index_num_samples(self):
        """
        Tests the number of samples seen in each class.
        """
        self.assertEqual(self.data.class_index_num_samples, [5, 5])

    def test_num_samples_rarest_class(self):
        """
        Tests the number of samples in the rarest class.
        """
        self.assertEqual(self.data.number_samples_in_rarest_class, 5)


class TestTrainDataset2(unittest.TestCase):
    """
    Tests loading dataset with its numeric attributes.
    """
    def setUp(self):
        """
        Loads dataset config.
        """
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset2'))

    def test_load_dataset(self):
        """
        Tests if correctly loads the dataset (this has a key_attrib_index).
        """
        dataset.Dataset(self.config["filepath"],
                        self.config["key attrib index"],
                        self.config["class attrib index"],
                        self.config["split char"],
                        self.config["missing value string"],
                        load_numeric=True)


class TestTrainDataset2Attributes(unittest.TestCase):
    """
    Tests loaded train Dataset attributes.
    """
    def setUp(self):
        """
        Loads dataset config and Dataset.
        """
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset2'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=True)

    def test_valid_attribs(self):
        """
        Tests which attributes are considered valid.
        """
        self.assertEqual(self.data.valid_nominal_attribute, [False, True, True, False, False])
        self.assertEqual(self.data.valid_numeric_attribute, [False, False, False, True, False])
        self.assertEqual(self.data.load_numeric, True)

    def test_num_samples(self):
        """
        Tests number of samples loaded.
        """
        self.assertEqual(self.data.num_samples, 10)

    def test_num_classes(self):
        """
        Tests number of classes seen.
        """
        self.assertEqual(self.data.num_classes, 3)

    def test_samples_classes(self):
        """
        Tests class seen for each sample.
        """
        self.assertEqual(self.data.sample_class, [0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

    def test_key_attrib_index(self):
        """
        Tests the key_attrib_index.
        """
        self.assertEqual(self.data.key_attrib_index, 0)

    def test_class_attrib_index(self):
        """
        Tests the class_attrib_index.
        """
        self.assertEqual(self.data.class_attrib_index, -1)

    def test_attrib_names(self):
        """
        Tests the attributes' names.
        """
        self.assertEqual(self.data.attrib_names,
                         ['key', 'nominal_attrib1', 'nominal_attrib2', 'numeric_attrib', 'class'])

    def test_class_index_num_samples(self):
        """
        Tests the number of samples seen in each class.
        """
        self.assertEqual(self.data.class_index_num_samples, [3, 3, 4])

    def test_num_samples_rarest_class(self):
        """
        Tests the number of samples in the rarest class.
        """
        self.assertEqual(self.data.number_samples_in_rarest_class, 3)


class TestWrongTestDataset2(unittest.TestCase):
    """
    Tests wether a test dataset with unmatching headers does raise ValueError.
    """
    def setUp(self):
        """
        Loads train dataset config, train Dataset and test dataset config.
        """
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset2'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=True)
        self.test_config = dataset.load_config(os.path.join(
            '.', 'data', 'wrong_test_dataset2'))

    def test_load_dataset(self):
        """
        Tests if correctly loads the dataset. Since headers don't match, should raise ValueError.
        """
        with self.assertRaises(ValueError):
            self.data.load_test_set_from_csv(self.test_config["filepath"],
                                             self.test_config["key attrib index"],
                                             self.test_config["class attrib index"],
                                             self.test_config["split char"],
                                             self.test_config["missing value string"])

class TestTestDataset2(unittest.TestCase):
    """
    Tests wether it correctly loads the test dataset.
    """
    def setUp(self):
        """
        Loads train dataset config, train Dataset and test dataset config.
        """
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset2'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=True)
        self.test_config = dataset.load_config(os.path.join(
            '.', 'data', 'test_dataset2'))

    def test_loading(self):
        """
        Tests if correctly loads the dataset.
        """
        self.data.load_test_set_from_csv(self.test_config["filepath"],
                                         self.test_config["key attrib index"],
                                         self.test_config["class attrib index"],
                                         self.test_config["split char"],
                                         self.test_config["missing value string"])


class TestTestDataset2Attributes(unittest.TestCase):
    """
    Tests loaded test Dataset attributes.
    """
    def setUp(self):
        """
        Loads train dataset config, train Dataset, test dataset config and test Dataset.
        """
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset2'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=True)
        self.test_config = dataset.load_config(os.path.join(
            '.', 'data', 'test_dataset2'))
        self.data.load_test_set_from_csv(self.test_config["filepath"],
                                         self.test_config["key attrib index"],
                                         self.test_config["class attrib index"],
                                         self.test_config["split char"],
                                         self.test_config["missing value string"])

    def test_test_num_samples(self):
        """
        Tests the number of test samples loaded.
        """
        self.assertEqual(self.data.test_num_samples, 10)

    def test_test_samples_classes(self):
        """
        Tests the class of each test sample loaded.
        """
        self.assertEqual(self.data.test_sample_class, [0, 0, 0, 1, 1, 1, 2, 2, 3, 3])


if __name__ == '__main__':
    unittest.main()
