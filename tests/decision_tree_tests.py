#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright Felipe de Albuquerque Mello Pereira 2017


"""
Module containing all tests for the dataset module.
"""


import os
import sys
import unittest

import numpy as np

sys.path.insert(0, '../src')
import dataset
import decision_tree


class TestCreateTreeNode(unittest.TestCase):
    """
    Tests creating TreeNode
    """
    def setUp(self):
        """
        Loads dataset config and Dataset with numeric attributes.
        """
        self.config = dataset.load_config(os.path.join(
            '.', 'data', 'train_dataset1'))
        self.data = dataset.Dataset(self.config["filepath"],
                                    self.config["key attrib index"],
                                    self.config["class attrib index"],
                                    self.config["split char"],
                                    self.config["missing value string"],
                                    load_numeric=False)

    def test_tree_node_creation(self):
        """
        Tests creation of TreeNode.
        """
        self.assertIsInstance(decision_tree.TreeNode(self.data,
                                                     list(range(self.data.num_samples)),
                                                     self.data.valid_nominal_attribute,
                                                     max_depth_remaining=0,
                                                     min_samples_per_node=1,
                                                     use_stop_conditions=False,
                                                     max_p_value_chi_sq=None),
                              decision_tree.TreeNode)


class TestTreeNodeAttributes(unittest.TestCase):
    """
    Tests creating TreeNode
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
        self.tree_node = decision_tree.TreeNode(self.data,
                                                list(range(self.data.num_samples)),
                                                self.data.valid_nominal_attribute,
                                                max_depth_remaining=0,
                                                min_samples_per_node=1,
                                                use_stop_conditions=False,
                                                max_p_value_chi_sq=None)

    def test_max_depth_remaining(self):
        """
        Tests TreeNode's max_depth_remaining.
        """
        self.assertEqual(self.tree_node.max_depth_remaining, 0)

    def test_is_leaf(self):
        """
        Tests TreeNode's is_leaf.
        """
        self.assertTrue(self.tree_node.is_leaf)

    def test_node_split(self):
        """
        Tests TreeNode's node_split.
        """
        self.assertIsNone(self.tree_node.node_split)

    def test_nodes(self):
        """
        Tests TreeNode's node_split.
        """
        self.assertIsInstance(self.tree_node.nodes, list)
        self.assertEqual(len(self.tree_node.nodes), 0)

    def test_curr_dataset(self):
        """
        Tests TreeNode's curr_dataset.
        """
        self.assertEqual(self.tree_node.curr_dataset, self.data)

    def test_valid_samples_indices(self):
        """
        Tests TreeNode's valid_samples_indices.
        """
        self.assertEqual(self.tree_node.valid_samples_indices, list(range(self.data.num_samples)))

    def test_valid_nominal_attribute(self):
        """
        Tests TreeNode's valid_nominal_attribute.
        """
        self.assertEqual(self.tree_node.valid_nominal_attribute, self.data.valid_nominal_attribute)

    def test_num_valid_samples(self):
        """
        Tests TreeNode's num_valid_samples.
        """
        self.assertEqual(self.tree_node.num_valid_samples, self.data.num_samples)

    def test_class_index_num_samples(self):
        """
        Tests TreeNode's class_index_num_samples.
        """
        self.assertEqual(self.tree_node.class_index_num_samples, self.data.class_index_num_samples)

    def test_number_non_empty_classes(self):
        """
        Tests TreeNode's number_non_empty_classes.
        """
        self.assertEqual(self.tree_node.number_non_empty_classes, self.data.num_classes)

    def test_most_common_int_class(self):
        """
        Tests TreeNode's most_common_int_class.
        """
        self.assertEqual(self.tree_node.most_common_int_class, 0)

    def test_contingency_tables(self):
        """
        Tests TreeNode's contingency_tables.
        """
        self.assertEqual(len(self.tree_node.contingency_tables),
                         len(self.data.valid_nominal_attribute))

        self.assertTrue(np.array_equal(self.tree_node.contingency_tables[0].contingency_table,
                                       np.array([[5, 0], [0, 5]], dtype=int)))
        self.assertTrue(np.array_equal(self.tree_node.contingency_tables[0].values_num_samples,
                                       np.array([5, 5], dtype=int)))

        self.assertEqual(self.tree_node.contingency_tables[1],
                         (None, None))

        self.assertTrue(np.array_equal(self.tree_node.contingency_tables[2].contingency_table,
                                       np.array([[2, 3], [3, 2]], dtype=int)))
        self.assertTrue(np.array_equal(self.tree_node.contingency_tables[2].values_num_samples,
                                       np.array([5, 5], dtype=int)))

        self.assertEqual(self.tree_node.contingency_tables[3],
                         (None, None))

    def test_get_most_popular_subtree(self):
        """
        Tests TreeNode's get_most_popular_subtree().
        """
        self.assertEqual(self.tree_node.get_most_popular_subtree(), self.data.num_samples)

    def test_get_num_nodes(self):
        """
        Tests TreeNode's get_num_nodes().
        """
        self.assertEqual(self.tree_node.get_num_nodes(), 1)

    def test_get_max_depth(self):
        """
        Tests TreeNode's get_max_depth().
        """
        self.assertEqual(self.tree_node.get_max_depth(), 0)

    def test_prune_trivial_subtrees(self):
        """
        Tests TreeNode's prune_trivial_subtrees().
        """
        self.assertEqual(self.tree_node.prune_trivial_subtrees(), 0)


class TestDecisionTree(unittest.TestCase):
    """
    Tests DecisionTree creation.
    """

    def setUp(self):
        """
        Imports criteria module and a criterion from it.
        """
        import criteria
        self.criterion = criteria.GiniGain

    def test_decision_tree_creation(self):
        """
        Tests DecisionTree creation.
        """
        self.assertIsInstance(decision_tree.DecisionTree(self.criterion),
                              decision_tree.DecisionTree)


class TestDecisionTreeAttributes(unittest.TestCase):
    """
    Tests DecisionTree attributes.
    """
    def setUp(self):
        """
        Loads dataset config and Dataset without numeric attributes.
        """
        import criteria
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

    def test_get_root_node(self):
        """
        Tests DecisionTree's get_root_node().
        """
        self.assertIsNone(self.decision_tree.get_root_node())

    def test_train(self):
        """
        Tests DecisionTree's train().
        """
        (time_taken_pruning,
         num_nodes_pruned) = self.decision_tree.train(self.data,
                                                      list(range(self.data.num_samples)),
                                                      max_depth=1,
                                                      min_samples_per_node=1,
                                                      use_stop_conditions=False,
                                                      max_p_value_chi_sq=None)
        self.assertIsInstance(time_taken_pruning, float)
        self.assertIsInstance(num_nodes_pruned, int)
        self.assertEqual(num_nodes_pruned, 0)

    def test_train_and_test(self):
        """
        Tests DecisionTree's train_and_test().
        """
        ((classifications,
          num_correct_classifications,
          num_correct_classifications_wo_unkown,
          total_cost,
          total_cost_wo_unkown,
          classified_with_unkown_value_array,
          num_unkown,
          unkown_value_attrib_index_array),
         max_depth_found,
         time_taken_pruning,
         num_nodes_pruned) = self.decision_tree.train_and_test(self.data,
                                                               list(range(self.data.num_samples)),
                                                               list(range(self.data.num_samples)),
                                                               max_depth=1,
                                                               min_samples_per_node=1,
                                                               use_stop_conditions=False,
                                                               max_p_value_chi_sq=None)
        self.assertEqual(classifications, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.assertEqual(num_correct_classifications, 10)
        self.assertEqual(num_correct_classifications_wo_unkown, 10)
        self.assertEqual(total_cost, 0.0)
        self.assertEqual(total_cost_wo_unkown, 0.0)
        self.assertEqual(classified_with_unkown_value_array, [False] * 10)
        self.assertEqual(num_unkown, 0)
        self.assertEqual(unkown_value_attrib_index_array, [None] * 10)
        self.assertEqual(max_depth_found, 1)
        self.assertIsInstance(time_taken_pruning, float)
        self.assertEqual(num_nodes_pruned, 0)

    def test_cross_validate_stratified(self):
        """
        Tests DecisionTree's stratified cross_validate().
        """

        # Fold 1:
        # training_samples_indices = [3, 4, 8, 9]
        # validation_samples_indices = [0, 1, 2, 5, 6, 7]
        #
        # Fold 2:
        # training_samples_indices = [0, 1, 2, 5, 6, 7]
        # validation_samples_indices = [3, 4, 8, 9]

        (classifications,
         num_correct_classifications,
         num_correct_classifications_wo_unkown,
         total_cost,
         total_cost_wo_unkown,
         classified_with_unkown_value_array,
         num_unkown,
         unkown_value_attrib_index_array,
         time_taken_pruning_per_fold,
         num_nodes_prunned_per_fold,
         max_depth_per_fold,
         num_nodes_per_fold,
         num_valid_nominal_attributes_in_root_per_fold,
         num_values_root_attribute_list,
         num_trivial_splits,
         trivial_accuracy) = self.decision_tree.cross_validate(self.data,
                                                               num_folds=2,
                                                               max_depth=1,
                                                               min_samples_per_node=1,
                                                               is_stratified=True,
                                                               print_tree=False,
                                                               seed=1,
                                                               use_stop_conditions=False,
                                                               max_p_value_chi_sq=None)
        self.assertEqual(classifications, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.assertEqual(num_correct_classifications, 10)
        self.assertEqual(num_correct_classifications_wo_unkown, 10)
        self.assertEqual(total_cost, 0.0)
        self.assertEqual(total_cost_wo_unkown, 0.0)
        self.assertEqual(classified_with_unkown_value_array, [False] * 10)
        self.assertEqual(num_unkown, 0)
        self.assertEqual(unkown_value_attrib_index_array, [None] * 10)
        self.assertEqual(len(time_taken_pruning_per_fold), 2)
        self.assertEqual(len(num_nodes_prunned_per_fold), 2)
        self.assertEqual(num_nodes_prunned_per_fold, [0, 0])
        self.assertEqual(len(max_depth_per_fold), 2)
        self.assertEqual(max_depth_per_fold, [1, 1])
        self.assertEqual(len(num_nodes_per_fold), 2)
        self.assertEqual(num_valid_nominal_attributes_in_root_per_fold, [2, 2])
        self.assertEqual(num_values_root_attribute_list, [2, 2])
        self.assertEqual(num_nodes_per_fold, [3, 3])
        self.assertEqual(num_trivial_splits, 0)
        self.assertEqual(trivial_accuracy, 50.0)


    def test_cross_validate_non_stratified(self):
        """
        Tests DecisionTree's non-stratified cross_validate().
        """

        # Fold 1:
        # training_samples_indices = [5, 6, 7, 8, 9]
        # validation_samples_indices = [0, 1, 2, 3, 4]
        #
        # Fold 2:
        # training_samples_indices = [0, 1, 2, 3, 4]
        # validation_samples_indices = [5, 6, 7, 8, 9]

        (classifications,
         num_correct_classifications,
         num_correct_classifications_wo_unkown,
         total_cost,
         total_cost_wo_unkown,
         classified_with_unkown_value_array,
         num_unkown,
         unkown_value_attrib_index_array,
         time_taken_pruning_per_fold,
         num_nodes_prunned_per_fold,
         max_depth_per_fold,
         num_nodes_per_fold,
         num_valid_nominal_attributes_in_root_per_fold,
         num_values_root_attribute_list,
         num_trivial_splits,
         trivial_accuracy) = self.decision_tree.cross_validate(self.data,
                                                               num_folds=2,
                                                               max_depth=1,
                                                               min_samples_per_node=1,
                                                               is_stratified=False,
                                                               print_tree=False,
                                                               seed=1,
                                                               use_stop_conditions=False,
                                                               max_p_value_chi_sq=None)
        self.assertEqual(classifications, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(num_correct_classifications, 0)
        self.assertEqual(num_correct_classifications_wo_unkown, 0)
        self.assertEqual(total_cost, 10.0)
        self.assertEqual(total_cost_wo_unkown, 10.0)
        self.assertEqual(classified_with_unkown_value_array, [False] * 10)
        self.assertEqual(num_unkown, 0)
        self.assertEqual(unkown_value_attrib_index_array, [None] * 10)
        self.assertEqual(len(time_taken_pruning_per_fold), 2)
        self.assertEqual(len(num_nodes_prunned_per_fold), 2)
        self.assertEqual(num_nodes_prunned_per_fold, [0, 0])
        self.assertEqual(len(max_depth_per_fold), 2)
        self.assertEqual(max_depth_per_fold, [0, 0])
        self.assertEqual(len(num_nodes_per_fold), 2)
        self.assertEqual(num_valid_nominal_attributes_in_root_per_fold, [2, 2])
        self.assertEqual(len(num_values_root_attribute_list), 0)
        self.assertEqual(num_nodes_per_fold, [1, 1])
        self.assertEqual(num_trivial_splits, 2)
        self.assertEqual(trivial_accuracy, 0.0)


class TestAfterTraining(unittest.TestCase):
    """
    Tests DecisionTree and TreeNode's attributes after training.
    """
    def setUp(self):
        """
        Loads dataset config and Dataset without numeric attributes, trains the tree.
        """
        import criteria
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
        self.decision_tree.train(self.data,
                                 list(range(self.data.num_samples)),
                                 max_depth=1,
                                 min_samples_per_node=1,
                                 use_stop_conditions=False,
                                 max_p_value_chi_sq=None)

    def test_test_from_csv(self):
        """
        Tests DecisionTree's test_from_csv().
        """
        test_config = dataset.load_config(os.path.join('.', 'data', 'train_dataset1'))
        (classifications,
         num_correct_classifications,
         num_correct_classifications_wo_unkown,
         total_cost,
         total_cost_wo_unkown,
         classified_with_unkown_value_array,
         num_unkown,
         unkown_value_attrib_index_array) = self.decision_tree.test_from_csv(
             test_config["filepath"],
             test_config["key attrib index"],
             test_config["class attrib index"],
             test_config["split char"],
             test_config["missing value string"])

        self.assertEqual(classifications, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        self.assertEqual(num_correct_classifications, 11)
        self.assertEqual(num_correct_classifications_wo_unkown, 11)
        self.assertEqual(total_cost, 1.0)
        self.assertEqual(total_cost_wo_unkown, 1.0)
        self.assertEqual(classified_with_unkown_value_array, [False] * 12)
        self.assertEqual(num_unkown, 0)
        self.assertEqual(unkown_value_attrib_index_array, [None] * 12)

    def test_max_depth_remaining(self):
        """
        Tests TreeNode's max_depth_remaining.
        """
        self.assertEqual(self.decision_tree.get_root_node().max_depth_remaining, 1)
        self.assertEqual(self.decision_tree.get_root_node().nodes[0].max_depth_remaining, 0)
        self.assertEqual(self.decision_tree.get_root_node().nodes[1].max_depth_remaining, 0)

    def test_is_leaf(self):
        """
        Tests TreeNode's is_leaf.
        """
        self.assertFalse(self.decision_tree.get_root_node().is_leaf)
        self.assertTrue(self.decision_tree.get_root_node().nodes[0].is_leaf)
        self.assertTrue(self.decision_tree.get_root_node().nodes[1].is_leaf)

    def test_node_split(self):
        """
        Tests TreeNode's node_split.
        """
        self.assertIsInstance(self.decision_tree.get_root_node().node_split,
                              decision_tree.NodeSplit)
        self.assertEqual(self.decision_tree.get_root_node().node_split.separation_attrib_index, 0)
        self.assertEqual(self.decision_tree.get_root_node().node_split.splits_values,
                         [set([0]), set([1])])
        self.assertEqual(self.decision_tree.get_root_node().node_split.values_to_split, {0: 0, 1:1})
        self.assertEqual(self.decision_tree.get_root_node().node_split.criterion_value, 0.5)

    def test_nodes(self):
        """
        Tests TreeNode's node_split.
        """
        self.assertEqual(len(self.decision_tree.get_root_node().nodes), 2)
        self.assertIsInstance(self.decision_tree.get_root_node().nodes[0], decision_tree.TreeNode)
        self.assertIsInstance(self.decision_tree.get_root_node().nodes[1], decision_tree.TreeNode)

    def test_valid_samples_indices(self):
        """
        Tests TreeNode's valid_samples_indices.
        """
        self.assertEqual(self.decision_tree.get_root_node().valid_samples_indices,
                         list(range(self.data.num_samples)))
        self.assertEqual(self.decision_tree.get_root_node().nodes[0].valid_samples_indices,
                         [0, 1, 2, 3, 4])
        self.assertEqual(self.decision_tree.get_root_node().nodes[1].valid_samples_indices,
                         [5, 6, 7, 8, 9])

    def test_num_valid_samples(self):
        """
        Tests TreeNode's num_valid_samples.
        """
        self.assertEqual(self.decision_tree.get_root_node().num_valid_samples,
                         self.data.num_samples)
        self.assertEqual(self.decision_tree.get_root_node().nodes[0].num_valid_samples, 5)
        self.assertEqual(self.decision_tree.get_root_node().nodes[1].num_valid_samples, 5)

    def test_class_index_num_samples(self):
        """
        Tests TreeNode's class_index_num_samples.
        """
        self.assertEqual(self.decision_tree.get_root_node().class_index_num_samples,
                         self.data.class_index_num_samples)
        self.assertEqual(self.decision_tree.get_root_node().nodes[0].class_index_num_samples,
                         [5, 0])
        self.assertEqual(self.decision_tree.get_root_node().nodes[1].class_index_num_samples,
                         [0, 5])

    def test_number_non_empty_classes(self):
        """
        Tests TreeNode's number_non_empty_classes.
        """
        self.assertEqual(self.decision_tree.get_root_node().number_non_empty_classes,
                         self.data.num_classes)
        self.assertEqual(self.decision_tree.get_root_node().nodes[0].number_non_empty_classes, 1)
        self.assertEqual(self.decision_tree.get_root_node().nodes[1].number_non_empty_classes, 1)

    def test_most_common_int_class(self):
        """
        Tests TreeNode's most_common_int_class.
        """
        self.assertEqual(self.decision_tree.get_root_node().most_common_int_class, 0)
        self.assertEqual(self.decision_tree.get_root_node().nodes[0].most_common_int_class, 0)
        self.assertEqual(self.decision_tree.get_root_node().nodes[1].most_common_int_class, 1)

    def test_get_most_popular_subtree(self):
        """
        Tests TreeNode's get_most_popular_subtree().
        """
        self.assertEqual(self.decision_tree.get_root_node().get_most_popular_subtree(), 5)
        self.assertEqual(self.decision_tree.get_root_node().nodes[0].get_most_popular_subtree(), 5)
        self.assertEqual(self.decision_tree.get_root_node().nodes[1].get_most_popular_subtree(), 5)

    def test_get_num_nodes(self):
        """
        Tests TreeNode's get_num_nodes().
        """
        self.assertEqual(self.decision_tree.get_root_node().get_num_nodes(), 3)

    def test_get_max_depth(self):
        """
        Tests TreeNode's get_max_depth().
        """
        self.assertEqual(self.decision_tree.get_root_node().get_max_depth(), 1)

    def test_prune_trivial_subtrees(self):
        """
        Tests TreeNode's prune_trivial_subtrees().
        """
        self.assertEqual(self.decision_tree.get_root_node().prune_trivial_subtrees(), 0)



if __name__ == '__main__':
    unittest.main()
