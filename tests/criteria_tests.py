#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright Felipe de Albuquerque Mello Pereira 2017


"""
Module containing all tests for the dataset module.
"""


import math
import os
import sys
import unittest

sys.path.insert(0, '../src')
import criteria
import dataset
import decision_tree


class TestGiniGainTwoClasses(unittest.TestCase):
    """
    Tests Gini Gain criterion using dataset with two classes.
    """
    def setUp(self):
        """
        Loads dataset config.
        """
        self.criterion = criteria.GiniGain
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset1'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=False)
        self.decision_tree = decision_tree.DecisionTree(self.criterion)

    def test_gini_gain(self):
        """
        Tests Gini Gain criterion using dataset with two classes.
        """
        self.decision_tree.train(self.data,
                                 list(range(self.data.num_samples)),
                                 max_depth=1,
                                 min_samples_per_node=1,
                                 use_stop_conditions=False,
                                 max_p_value_chi_sq=None)
        self.assertEqual(self.decision_tree.get_root_node().node_split.separation_attrib_index, 0)
        self.assertEqual(self.decision_tree.get_root_node().node_split.splits_values,
                         [set([0]), set([1])])
        self.assertEqual(self.decision_tree.get_root_node().node_split.values_to_split,
                         {0:0, 1:1})
        self.assertEqual(self.decision_tree.get_root_node().node_split.criterion_value, 0.5)


class TestGiniGainMoreClasses(unittest.TestCase):
    """
    Tests Gini Gain criterion using dataset with more than two classes.
    """
    def setUp(self):
        """
        Loads dataset config.
        """
        self.criterion = criteria.GiniGain
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset2'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=False)
        self.decision_tree = decision_tree.DecisionTree(self.criterion)

    def test_gini_gain(self):
        """
        Tests Gini Gain criterion using dataset with more than two classes.
        """
        self.decision_tree.train(self.data,
                                 list(range(self.data.num_samples)),
                                 max_depth=1,
                                 min_samples_per_node=1,
                                 use_stop_conditions=False,
                                 max_p_value_chi_sq=None)
        self.assertEqual(self.decision_tree.get_root_node().node_split.separation_attrib_index, 1)
        self.assertEqual(self.decision_tree.get_root_node().node_split.splits_values,
                         [set([0]), set([1]), set([2])])
        self.assertEqual(self.decision_tree.get_root_node().node_split.values_to_split,
                         {0:0, 1:1, 2:2})
        self.assertEqual(self.decision_tree.get_root_node().node_split.criterion_value, 0.66)


class TestTwoingTwoClasses(unittest.TestCase):
    """
    Tests Twoing criterion using dataset with two classes.
    """
    def setUp(self):
        """
        Loads dataset config.
        """
        self.criterion = criteria.Twoing
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset1'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=False)
        self.decision_tree = decision_tree.DecisionTree(self.criterion)

    def test_twoing(self):
        """
        Tests Twoing criterion using dataset with two classes.
        """
        self.decision_tree.train(self.data,
                                 list(range(self.data.num_samples)),
                                 max_depth=1,
                                 min_samples_per_node=1,
                                 use_stop_conditions=False,
                                 max_p_value_chi_sq=None)
        self.assertEqual(self.decision_tree.get_root_node().node_split.separation_attrib_index, 0)
        self.assertEqual(self.decision_tree.get_root_node().node_split.splits_values,
                         [set([0]), set([1])])
        self.assertEqual(self.decision_tree.get_root_node().node_split.values_to_split,
                         {0:0, 1:1})
        self.assertEqual(self.decision_tree.get_root_node().node_split.criterion_value, 0.5)


class TestTwoingMoreClasses(unittest.TestCase):
    """
    Tests Twoing criterion using dataset with more than two classes.
    """
    def setUp(self):
        """
        Loads dataset config.
        """
        self.criterion = criteria.Twoing
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset2'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=False)
        self.decision_tree = decision_tree.DecisionTree(self.criterion)

    def test_twoing(self):
        """
        Tests Twoing criterion using dataset with more than two classes.
        """
        self.decision_tree.train(self.data,
                                 list(range(self.data.num_samples)),
                                 max_depth=1,
                                 min_samples_per_node=1,
                                 use_stop_conditions=False,
                                 max_p_value_chi_sq=None)
        self.assertEqual(self.decision_tree.get_root_node().node_split.separation_attrib_index, 1)
        self.assertEqual(self.decision_tree.get_root_node().node_split.splits_values,
                         [set([2]), set([0, 1])])
        self.assertEqual(self.decision_tree.get_root_node().node_split.values_to_split,
                         {0:1, 1:1, 2:0})
        self.assertEqual(self.decision_tree.get_root_node().node_split.criterion_value, 0.48)


class TestGainRatioTwoClasses(unittest.TestCase):
    """
    Tests Gain Ratio criterion using dataset with two classes.
    """
    def setUp(self):
        """
        Loads dataset config.
        """
        self.criterion = criteria.GainRatio
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset1'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=False)
        self.decision_tree = decision_tree.DecisionTree(self.criterion)

    def test_gain_ratio(self):
        """
        Tests Gain Ratio criterion using dataset with two classes.
        """
        self.decision_tree.train(self.data,
                                 list(range(self.data.num_samples)),
                                 max_depth=1,
                                 min_samples_per_node=1,
                                 use_stop_conditions=False,
                                 max_p_value_chi_sq=None)
        self.assertEqual(self.decision_tree.get_root_node().node_split.separation_attrib_index, 0)
        self.assertEqual(self.decision_tree.get_root_node().node_split.splits_values,
                         [set([0]), set([1])])
        self.assertEqual(self.decision_tree.get_root_node().node_split.values_to_split,
                         {0:0, 1:1})
        self.assertEqual(self.decision_tree.get_root_node().node_split.criterion_value, 1.0)


class TestGainRatioMoreClasses(unittest.TestCase):
    """
    Tests Gain Ratio criterion using dataset with more than two classes.
    """
    def setUp(self):
        """
        Loads dataset config.
        """
        self.criterion = criteria.GainRatio
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset2'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=False)
        self.decision_tree = decision_tree.DecisionTree(self.criterion)

    def test_gain_ratio(self):
        """
        Tests Gain Ratio criterion using dataset with more than two classes.
        """
        self.decision_tree.train(self.data,
                                 list(range(self.data.num_samples)),
                                 max_depth=1,
                                 min_samples_per_node=1,
                                 use_stop_conditions=False,
                                 max_p_value_chi_sq=None)
        self.assertEqual(self.decision_tree.get_root_node().node_split.separation_attrib_index, 1)
        self.assertEqual(self.decision_tree.get_root_node().node_split.splits_values,
                         [set([0]), set([1]), set([2])])
        self.assertEqual(self.decision_tree.get_root_node().node_split.values_to_split,
                         {0:0, 1:1, 2:2})
        self.assertEqual(self.decision_tree.get_root_node().node_split.criterion_value, 1.0)




class TestInformationGainTwoClasses(unittest.TestCase):
    """
    Tests Information Gain criterion using dataset with two classes.
    """
    def setUp(self):
        """
        Loads dataset config.
        """
        self.criterion = criteria.InformationGain
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset1'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=False)
        self.decision_tree = decision_tree.DecisionTree(self.criterion)

    def test_information_gain(self):
        """
        Tests Information Gain criterion using dataset with two classes.
        """
        self.decision_tree.train(self.data,
                                 list(range(self.data.num_samples)),
                                 max_depth=1,
                                 min_samples_per_node=1,
                                 use_stop_conditions=False,
                                 max_p_value_chi_sq=None)
        self.assertEqual(self.decision_tree.get_root_node().node_split.separation_attrib_index, 0)
        self.assertEqual(self.decision_tree.get_root_node().node_split.splits_values,
                         [set([0]), set([1])])
        self.assertEqual(self.decision_tree.get_root_node().node_split.values_to_split,
                         {0:0, 1:1})
        self.assertEqual(self.decision_tree.get_root_node().node_split.criterion_value, 1.0)


class TestInformationGainMoreClasses(unittest.TestCase):
    """
    Tests Information Gain criterion using dataset with more than two classes.
    """
    def setUp(self):
        """
        Loads dataset config.
        """
        self.criterion = criteria.InformationGain
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset2'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=False)
        self.decision_tree = decision_tree.DecisionTree(self.criterion)

    def test_information_gain(self):
        """
        Tests Information Gain criterion using dataset with more than two classes.
        """
        self.decision_tree.train(self.data,
                                 list(range(self.data.num_samples)),
                                 max_depth=1,
                                 min_samples_per_node=1,
                                 use_stop_conditions=False,
                                 max_p_value_chi_sq=None)
        self.assertEqual(self.decision_tree.get_root_node().node_split.separation_attrib_index, 1)
        self.assertEqual(self.decision_tree.get_root_node().node_split.splits_values,
                         [set([0]), set([1]), set([2])])
        self.assertEqual(self.decision_tree.get_root_node().node_split.values_to_split,
                         {0:0, 1:1, 2:2})
        self.assertAlmostEqual(self.decision_tree.get_root_node().node_split.criterion_value,
                               2. * -0.3 * math.log2(0.3) - 0.4 * math.log2(0.4))


if __name__ == '__main__':
    unittest.main()
