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
    def test_nonexistent_file(self):
        assert dataset.load_config('.') is None

    def test_missing_name(self):
        assert dataset.load_config(os.path.join('.', 'data', 'wrong_configs', 'config_1')) is None

    def test_missing_key_index(self):
        assert dataset.load_config(os.path.join('.', 'data', 'wrong_configs', 'config_2')) is None

    def test_missing_class_index(self):
        assert dataset.load_config(os.path.join('.', 'data', 'wrong_configs', 'config_3')) is None

    def test_missing_split_char(self):
        assert dataset.load_config(os.path.join('.', 'data', 'wrong_configs', 'config_4')) is None

    def test_missing_missing_value_str(self):
        assert dataset.load_config(os.path.join('.', 'data', 'wrong_configs', 'config_5')) is None


class TestDatasetWithoutValidAttribs(unittest.TestCase):
    def setUp(self):
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'dataset_without_valid_attributes'))

    def test_no_valid_attrib(self):
        with self.assertRaises(ValueError):
            dataset.Dataset(self.config["filepath"],
                            self.config["key attrib index"],
                            self.config["class attrib index"],
                            self.config["split char"],
                            self.config["missing value string"],
                            load_numeric=True)


class TestTrainDataset1WithoutNumeric(unittest.TestCase):
    def setUp(self):
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset1'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=False)

    def test_valid_attribs(self):
        self.assertEqual(self.data.valid_nominal_attribute, [True, True, False, False])
        self.assertEqual(self.data.valid_numeric_attribute, [False] * 4)
        self.assertEqual(self.data.load_numeric, False)


class TestTrainDataset1WithoutNumeric(unittest.TestCase):
    def setUp(self):
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset1'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=True)

    def test_valid_attribs(self):
        self.assertEqual(self.data.valid_nominal_attribute, [True, True, False, False])
        self.assertEqual(self.data.valid_numeric_attribute, [False, False, True, False])
        self.assertEqual(self.data.load_numeric, True)

    def test_num_samples(self):
        self.assertEqual(self.data.num_samples, 10)

    def test_num_classes(self):
        self.assertEqual(self.data.num_classes, 2)

    def test_samples_classes(self):
        self.assertEqual(self.data.sample_class, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    def test_key_attrib_index(self):
        assert self.data.key_attrib_index is None

    def test_class_attrib_index(self):
        self.assertEqual(self.data.class_attrib_index, -1)

    def test_attrib_names(self):
        self.assertEqual(self.data.attrib_names,
                         ['nominal_attrib1', 'nominal_attrib2', 'numeric_attrib', 'class'])

    def test_class_index_num_samples(self):
        self.assertEqual(self.data.class_index_num_samples, [5, 5])

    def test_num_samples_rarest_class(self):
        self.assertEqual(self.data.number_samples_in_rarest_class, 5)


if __name__ == '__main__':
    unittest.main()
