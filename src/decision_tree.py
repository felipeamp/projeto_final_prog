#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright Felipe de Albuquerque Mello Pereira 2017

"""
Module containing the DecisionTree, TreeNode and NodeSplit classes.
"""

import collections
import math
import random
import sys
import timeit

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import chi2


#: Minimum number of samples needed in the two most frequent values of an attribute such that it is
#: considered valid.
MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE = 40

#: Use the minimum number of samples in the two most frequent classes as a criteria to allow a split
#: or not.
USE_MIN_SAMPLES_SECOND_LARGEST_CLASS = True
#: Minimum number of samples needed in the two most frequent classes such that this node can be
#: split.
MIN_SAMPLES_SECOND_LARGEST_CLASS = 40

#: Contains the information about an attribute's contingency table. When empty, defaults to
#: `(None, None)`. Otherwise, contains a np.array in each entry.
ContingencyTable = collections.namedtuple('ContingencyTable',
                                          ['contingency_table',
                                           'values_num_samples'])
ContingencyTable.__new__.__defaults__ = (None, None)


class DecisionTree(object):
    """
    Data structure containing basic information pertaining to the whole tree.

    This class' state should be accessed only indirectly, through its methods.

    :param criteria.Criterion criterion: criterion which will be used to generate the tree
        nodes/splits.
    """
    def __init__(self, criterion):
        self._criterion = criterion
        self._curr_dataset = None
        self._root_node = None

    def get_root_node(self):
        """
        :return: the TreeNode at the root of the tree. Might be None.
        :rtype: decision_tree.TreeNode or None
        """
        return self._root_node

    def get_trivial_accuracy(self, test_samples_indices):
        """
        Gets the accuracy obtained by classifying every test samples in the most common class
        among training samples. Must be called after training the tree.

        :param test_samples_indices: indices of samples to be tested.
        :type test_samples_indices: list[int]
        :return: the accuracy percentage.
        :rtype: float.
        """
        num_correct = sum(self._curr_dataset.sample_class[curr_sample_index]
                          == self._root_node.most_common_int_class
                          for curr_sample_index in test_samples_indices)
        return 100.0 * num_correct / len(test_samples_indices)

    def _classify_sample(self, sample, sample_key):
        if self._root_node is None:
            print('Cannot classify in untrained tree!')
            sys.exit(1)
        curr_node = self._root_node
        classified_with_unkown_value = False
        unkown_value_attrib_index = None
        while not curr_node.is_leaf:
            split_attrib_index = curr_node.node_split.separation_attrib_index
            sample_value = sample[split_attrib_index]
            try:
                split_index = curr_node.node_split.values_to_split[sample_value]
                curr_node = curr_node.nodes[split_index]
            except KeyError:
                print('\tSample {} has value unkown to split'
                      ' (value = {} in attrib #{}).'.format(
                          sample_key,
                          sample_value,
                          curr_node.node_split.separation_attrib_index))
                classified_with_unkown_value = True
                unkown_value_attrib_index = curr_node.node_split.separation_attrib_index
                break
        return (curr_node.most_common_int_class,
                classified_with_unkown_value,
                unkown_value_attrib_index)

    def _classify_samples(self, test_dataset_sample, test_dataset_sample_class,
                          test_dataset_sample_costs, test_samples_indices,
                          test_dataset_sample_keys):
        print('Starting classifications...')
        classifications = []
        classified_with_unkown_value_array = []
        unkown_value_attrib_index_array = []

        num_correct_classifications = 0
        num_correct_classifications_wo_unkown = 0
        num_unkown = 0

        total_cost = 0.0
        total_cost_wo_unkown = 0.0

        for test_sample_index in test_samples_indices:
            (predicted_class,
             classified_with_unkown_value,
             unkown_value_attrib_index) = self._classify_sample(
                 test_dataset_sample[test_sample_index],
                 test_dataset_sample_keys[test_sample_index])
            classifications.append(predicted_class)
            classified_with_unkown_value_array.append(classified_with_unkown_value)
            unkown_value_attrib_index_array.append(unkown_value_attrib_index)
            if predicted_class == test_dataset_sample_class[test_sample_index]:
                num_correct_classifications += 1
                if not classified_with_unkown_value:
                    num_correct_classifications_wo_unkown += 1
            total_cost += test_dataset_sample_costs[test_sample_index][predicted_class]
            if not classified_with_unkown_value:
                total_cost_wo_unkown += test_dataset_sample_costs[
                    test_sample_index][predicted_class]
            else:
                num_unkown += 1
        print('Done!')
        return (classifications,
                num_correct_classifications,
                num_correct_classifications_wo_unkown,
                total_cost,
                total_cost_wo_unkown,
                classified_with_unkown_value_array,
                num_unkown,
                unkown_value_attrib_index_array)

    def train(self, curr_dataset, training_samples_indices, max_depth, min_samples_per_node,
              use_stop_conditions=False, max_p_value_chi_sq=0.1):
        """
        Trains the tree in a recursive fashion, starting at the root's TreeNode. Afterwards,
        prunes the trivial subtrees.

        :param dataset.Dataset curr_dataset: dataset containing the samples used for training.
        :param training_samples_indices: list containing the indices of samples of `curr_dataset`
            used for training.
        :type training_samples_indices: list[int]
        :param int max_depth: maximum tree depth allowed. Zero means the root is a leaf.
        :param int min_samples_per_node: if a node has less than this number of training samples,
            it will necessarily be a leaf.
        :param bool use_stop_conditions: informs wether we should use pruning techniques to avoid
            using attributes with small number of samples (and, thus, avoiding statistical
            anomalies). An attribute will be considered invalid if it contains less than
            `MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE` samples in the second most frequent value
            (this way at least two values have this number of samples) or if a chi-square test,
            applied on the attributes' contingency table has a p-value greater or equal to
            `max_p_value_chi_sq`. When an attribute is considered invalid for the number of samples
            in the second most frequent value, it will be considered invalid in every child node of
            the current TreeNode. If it was considered invalid because of the chi-square test, it
            can be considered valid in a descendant node. Defaults to `False`.
        :param float max_p_value_chi_sq: is the maximum p-value allowed for an attribute to be
            accepted when doing chi-square tests (that is, when `use_stop_conditions` is `True`).A
            p-value of 1.0 is equal to 100%. Defaults to `0.1` (which is 10%).
        :return: tuple containing the time spent pruning the trained tree and the number of nodes
            prunned.
        :rtype: tuple(float, int)
        """
        self._curr_dataset = curr_dataset
        print('Starting tree training...')
        self._root_node = TreeNode(
            curr_dataset,
            training_samples_indices,
            curr_dataset.valid_nominal_attribute[:],
            max_depth,
            min_samples_per_node,
            use_stop_conditions,
            max_p_value_chi_sq)
        self._root_node.create_subtree(self._criterion)
        print('Starting pruning trivial subtrees...')
        start_time = timeit.default_timer()
        num_nodes_prunned = self._root_node.prune_trivial_subtrees()
        time_taken_pruning = timeit.default_timer() - start_time
        print('Done!')
        return time_taken_pruning, num_nodes_prunned

    def train_and_test(self, curr_dataset, training_samples_indices, validation_sample_indices,
                       max_depth, min_samples_per_node, use_stop_conditions=False,
                       max_p_value_chi_sq=0.1):
        """
        Trains a tree with part of the dataset (training samples) and tests the tree
        classification in another part (validation samples).

        Note that although the training and test samples are part of the same Dataset class, they
        usually shouldn't intersect.

        :param dataset.Dataset curr_dataset: dataset containing the samples used for training.
        :param training_samples_indices: list containing the indices of samples of `curr_dataset`
            used for training.
        :type training_samples_indices: list[int]
        :param validation_sample_indices: list containing the indices of samples of `curr_dataset`
            used to test the tree classification.
        :type validation_sample_indices: list[int]
        :param int max_depth: maximum tree depth allowed. Zero means the root is a leaf.
        :param int min_samples_per_node: if a node has less than this number of training samples,
            it will necessarily be a leaf.
        :param bool use_stop_conditions: informs wether we should use pruning techniques to avoid
            using attributes with small number of samples (and, thus, avoiding statistical
            anomalies). An attribute will be considered invalid if it contains less than
            `MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE` samples in the second most frequent value
            (this way at least two values have this number of samples) or if a chi-square test,
            applied on the attributes' contingency table has a p-value greater or equal to
            `max_p_value_chi_sq`. When an attribute is considered invalid for the number of samples
            in the second most frequent value, it will be considered invalid in every child node of
            the current TreeNode. If it was considered invalid because of the chi-square test, it
            can be considered valid in a descendant node. Defaults to `False`.
        :param float max_p_value_chi_sq: is the maximum p-value allowed for an attribute to be
            accepted when doing chi-square tests (that is, when `use_stop_conditions` is `True`). A
            p-value of 1.0 is equal to 100%. Defaults to `0.1` (which is 10%).
        :return: A tuple containing the tree's max depth in the second entry, the time taken
            pruning in the third entry and the number of nodes prunned in the fourth entry. In the
            first entry it returns another tuple containing, in order:

            * a list of predicted class for each validation sample;
            * the number of correct classifications;
            * the number of correct classifications done without validation samples with unkown
              values (that is, values that are unkown at a TreeNode -- they are classified as the
              most common class at that node);
            * the total cost of the classification errors (when errors costs are uniform, this is
              equal to the total number of validation samples minus the number of correct
              classifications);
            *  the total cost of the classification errors without considering validation samples
               with unkown values;
            * a list of booleans indicating if the i-th validation sample was classified with an
              unkown value;
            * the number of validation samples classified with unkown values;
            * list where the i-th entry has the attribute index used for classification of the i-th
              sample when an unkown value occurred.

        :rtype: tuple(tuple(list[int], int, int, float, float, list[bool], int, list[int]), int,
            float, int)
        """
        time_taken_pruning, num_nodes_prunned = self.train(curr_dataset,
                                                           training_samples_indices,
                                                           max_depth,
                                                           min_samples_per_node,
                                                           use_stop_conditions,
                                                           max_p_value_chi_sq)
        max_depth = self.get_root_node().get_max_depth()
        return (self._classify_samples(self._curr_dataset.samples,
                                       self._curr_dataset.sample_class,
                                       self._curr_dataset.sample_costs,
                                       validation_sample_indices,
                                       self._curr_dataset.sample_index_to_key),
                max_depth,
                time_taken_pruning,
                num_nodes_prunned)


    def cross_validate(self, curr_dataset, num_folds, max_depth, min_samples_per_node,
                       is_stratified=True, print_tree=False, seed=None, print_samples=False,
                       use_stop_conditions=False, max_p_value_chi_sq=0.1):
        """
        Does a cross-validation using a given dataset.

        It splits this dataset in `num_folds` folds and calls `train_and_test` on each. Might
        be given a seed for the dataset's random splitting and might be stratified.

        :param dataset.Dataset curr_dataset: dataset containing the samples used for training.
        :param int num_folds: number of folds used in the cross-validation.
        :param int max_depth: maximum tree depth allowed. Zero means the root is a leaf.
        :param int min_samples_per_node: if a node has less than this number of training samples,
            it will necessarily be a leaf.
        :param bool is_stratified: Indicates wheter the cross-validation should be stratified or
            just a simple k-fold cross-validation. Stratified means the samples' splitting will try
            to keep the classes' distribution the same across folds. Defaults to `True`.
        :param bool print_tree: Indicates wether the trees of every fold should be printed to
            stdout. Defaults to `False`.
        :param seed: indicates the seed that should be used to generate the random samples'
            splitting in folds. If `None`, a random seed is used. Defaults to `None`.
        :type seed: int or None
        :param bool print_samples: if `True`, prints the samples indices used at each fold. Used for
            debugging. Defaults to `False`.
        :param bool use_stop_conditions: informs wether we should use pruning techniques to avoid
            using attributes with small number of samples (and, thus, avoiding statistical
            anomalies). An attribute will be considered invalid if it contains less than
            `MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE` samples in the second most frequent value
            (this way at least two values have this number of samples) or if a chi-square test,
            applied on the attributes' contingency table has a p-value greater or equal to
            `max_p_value_chi_sq`. When an attribute is considered invalid for the number of samples
            in the second most frequent value, it will be considered invalid in every child node of
            the current TreeNode. If it was considered invalid because of the chi-square test, it
            can be considered valid in a descendant node. Note that numeric attributes are never
            tested in this way. Defaults to `False`.
        :param float max_p_value_chi_sq: is the maximum p-value allowed for an attribute to be
            accepted when doing chi-square tests (that is, when `use_stop_conditions` is `True`). A
            p-value of 1.0 is equal to 100%. Defaults to `0.1` (which is 10%).
        :return: A tuple containing, in order:

            * a list of predicted class for each sample;
            * the number of correct classifications;
            * the number of correct classifications done without samples with unkown values; (that
              is, values that are unkown at a TreeNode -- they are classified as the most common
              class at that node);
            * the total cost of the classification errors (when errors costs are uniform, this
              is equal to the total number of samples minus the number of correct classifications);
            * the total cost of the classification errors without considering samples with unkown
              values;
            * a list of booleans indicating if the i-th sample was classified with an unkown value;
            * the number of samples classified with unkown values;
            * list where the i-th entry has the attribute index used for classification of the i-th
              sample when an unkown value occurred;
            * list containing the time spent pruning in each fold;
            * list containing the number of nodes prunned in each fold;
            * list containing the maximum tree depth for each fold;
            * list containing the number of nodes per fold, after pruning;
            * list containing the number of valid attributes in root node in each fold;
            * Accuracy percentage obtained by classifying, in each fold, the test samples in the
              most common class among training samples.

        :rtype: tuple(list[int], int, int, float, float, list[bool], int, list[int], list[float],
            list[int], list[int], list[int], list[int], list[float])
        """
        classifications = [0] * curr_dataset.num_samples
        num_correct_classifications = 0
        num_correct_classifications_wo_unkown = 0
        total_cost = 0.0
        total_cost_wo_unkown = 0.0
        classified_with_unkown_value_array = [False] * curr_dataset.num_samples
        num_unkown = 0
        unkown_value_attrib_index_array = [0] * curr_dataset.num_samples
        max_depth_per_fold = []
        num_nodes_per_fold = []
        num_valid_nominal_attributes_in_root_per_fold = []
        num_values_root_attribute_list = []
        num_trivial_splits = 0
        time_taken_pruning_per_fold = []
        num_nodes_prunned_per_fold = []
        num_correct_trivial_classifications = 0

        fold_count = 0

        sample_indices_and_classes = list(enumerate(curr_dataset.sample_class))
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        random.shuffle(sample_indices_and_classes)
        shuffled_sample_indices, shuffled_sample_classes = zip(*sample_indices_and_classes)

        if is_stratified:
            for (training_randomized_indices,
                 validation_randomized_indices) in StratifiedKFold(n_splits=num_folds).split(
                     shuffled_sample_indices,
                     shuffled_sample_classes):

                training_samples_indices = [shuffled_sample_indices[index]
                                            for index in training_randomized_indices]
                validation_sample_indices = [shuffled_sample_indices[index]
                                             for index in validation_randomized_indices]

                if print_samples:
                    print('Samples used for validation in this fold:')
                    print(validation_sample_indices)
                    print()

                ((curr_classifications,
                  curr_num_correct_classifications,
                  curr_num_correct_classifications_wo_unkown,
                  curr_total_cost,
                  curr_total_cost_wo_unkown,
                  curr_classified_with_unkown_value_array,
                  curr_num_unkown,
                  curr_unkown_value_attrib_index_array),
                 curr_max_depth,
                 curr_time_taken_pruning,
                 curr_num_nodes_prunned) = self.train_and_test(curr_dataset,
                                                               training_samples_indices,
                                                               validation_sample_indices,
                                                               max_depth,
                                                               min_samples_per_node,
                                                               use_stop_conditions,
                                                               max_p_value_chi_sq)
                max_depth_per_fold.append(curr_max_depth)
                num_nodes_per_fold.append(self.get_root_node().get_num_nodes())
                num_valid_nominal_attributes_in_root_per_fold.append(
                    sum(self._root_node.valid_nominal_attribute))
                try:
                    root_node_split_attrib = self.get_root_node().node_split.separation_attrib_index
                    if curr_dataset.valid_nominal_attribute[root_node_split_attrib]:
                        num_values_root_attribute_list.append(sum(
                            num_samples > 0
                            for num_samples in self.get_root_node().contingency_tables[
                                root_node_split_attrib].values_num_samples))
                except AttributeError:
                    num_trivial_splits += 1
                for curr_index, validation_sample_index in enumerate(validation_sample_indices):
                    classifications[validation_sample_index] = curr_classifications[curr_index]
                    classified_with_unkown_value_array[validation_sample_index] = (
                        curr_classified_with_unkown_value_array[curr_index])
                    unkown_value_attrib_index_array[validation_sample_index] = (
                        curr_unkown_value_attrib_index_array[curr_index])
                num_correct_classifications += curr_num_correct_classifications
                num_correct_classifications_wo_unkown += curr_num_correct_classifications_wo_unkown
                total_cost += curr_total_cost
                total_cost_wo_unkown += curr_total_cost_wo_unkown
                num_unkown += curr_num_unkown
                num_correct_trivial_classifications += round(
                    len(validation_sample_indices) *
                    (self.get_trivial_accuracy(validation_sample_indices) / 100.0))

                fold_count += 1
                time_taken_pruning_per_fold.append(curr_time_taken_pruning)
                num_nodes_prunned_per_fold.append(curr_num_nodes_prunned)

                if print_tree:
                    print()
                    print('-' * 50)
                    print('Fold:', fold_count)
                    self.save_tree()
        else:
            for (training_samples_indices,
                 validation_sample_indices) in KFold(n_splits=num_folds).split(
                     shuffled_sample_indices):

                ((curr_classifications,
                  curr_num_correct_classifications,
                  curr_num_correct_classifications_wo_unkown,
                  curr_total_cost,
                  curr_total_cost_wo_unkown,
                  curr_classified_with_unkown_value_array,
                  curr_num_unkown,
                  curr_unkown_value_attrib_index_array),
                 curr_max_depth,
                 curr_time_taken_pruning,
                 curr_num_nodes_prunned) = self.train_and_test(curr_dataset,
                                                               training_samples_indices,
                                                               validation_sample_indices,
                                                               max_depth,
                                                               min_samples_per_node,
                                                               use_stop_conditions,
                                                               max_p_value_chi_sq)
                max_depth_per_fold.append(curr_max_depth)
                num_nodes_per_fold.append(self.get_root_node().get_num_nodes())
                num_valid_nominal_attributes_in_root_per_fold.append(
                    sum(self._root_node.valid_nominal_attribute))
                try:
                    root_node_split_attrib = self.get_root_node().node_split.separation_attrib_index
                    if curr_dataset.valid_nominal_attribute[root_node_split_attrib]:
                        num_values_root_attribute_list.append(sum(
                            num_samples > 0
                            for num_samples in self.get_root_node().contingency_tables[
                                root_node_split_attrib].values_num_samples))
                except AttributeError:
                    num_trivial_splits += 1
                for curr_index, validation_sample_index in enumerate(validation_sample_indices):
                    classifications[validation_sample_index] = curr_classifications[curr_index]
                    classified_with_unkown_value_array[validation_sample_index] = (
                        curr_classified_with_unkown_value_array[curr_index])
                    unkown_value_attrib_index_array[validation_sample_index] = (
                        curr_unkown_value_attrib_index_array[curr_index])
                num_correct_classifications += curr_num_correct_classifications
                num_correct_classifications_wo_unkown += curr_num_correct_classifications_wo_unkown
                total_cost += curr_total_cost
                total_cost_wo_unkown += curr_total_cost_wo_unkown
                num_unkown += curr_num_unkown
                num_correct_trivial_classifications += round(
                    len(validation_sample_indices) *
                    (self.get_trivial_accuracy(validation_sample_indices) / 100.0))

                fold_count += 1
                time_taken_pruning_per_fold.append(curr_time_taken_pruning)
                num_nodes_prunned_per_fold.append(curr_num_nodes_prunned)

                if print_tree:
                    print()
                    print('-' * 50)
                    print('Fold:', fold_count)
                    self.save_tree()

        return  (classifications,
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
                 100.0 * num_correct_trivial_classifications / curr_dataset.num_samples)

    def test(self, test_sample_indices):
        """
        Tests the (already trained) tree over samples from the same dataset as the
            training set. If the tree hasn't been trained, the program will exit.

        :param test_sample_indices: list of the test set indices for samples from the same dataset
            used for training.
        :type test_sample_indices: list[int]

        :return: A tuple containing, in order:

            * a list of predicted class for each test sample;
            * the number of correct classifications;
            * the number of correct classifications done without test samples with unkown values
              (that is, values that are unkown at a TreeNode -- they are classified as the most
              common class at that node);
            * the total cost of the classification errors (when errors costs are uniform, this is
              equal to the total number of test samples minus the number of correct
              classifications);
            *  the total cost of the classification errors without considering test samples with
               unkown values;
            * a list of booleans indicating if the i-th test sample was classified with an unkown
              value;
            * the number of test samples classified with unkown values;
            * list where the i-th entry has the attribute index used for classification of the i-th
              sample when an unkown value occurred.

        :rtype: tuple(list[int], int, int, float, float, list[bool], int, list[int])
        """

        if self._root_node is None:
            print('Decision tree must be trained before testing.')
            sys.exit(1)
        return self._classify_samples(self._curr_dataset.samples,
                                      self._curr_dataset.sample_class,
                                      self._curr_dataset.sample_costs,
                                      test_sample_indices,
                                      self._curr_dataset.sample_index_to_key)

    def test_from_csv(self, test_dataset_csv_filepath, key_attrib_index, class_attrib_index,
                      split_char, missing_value_string):
        """
        Tests the (already trained) tree using all samples from a given csv file. If the tree
        hasn't been trained, the program will exit.

        :param str test_dataset_csv_filepath: path to the test dataset.
        :param key_attrib_index: column index of the samples' keys on the csv.
        :type key_attrib_index: int or None
        :param int class_attrib_index: column index of the samples' classes on the csv.
        :param str split_char: char used to split columns in the csv.
        :param missing_value_string: string used to indicate that a sample does not have a value.
        :type missing_value_string: int or None

        :return: A tuple containing, in order:

            * a list of predicted class for each test sample;
            * the number of correct classifications;
            * the number of correct classifications done without test samples with unkown values
              (that is, values that are unkown at a TreeNode -- they are classified as the most
              common class at that node);
            * the total cost of the classification errors (when errors costs are uniform, this is
              equal to the total number of test samples minus the number of correct
              classifications);
            *  the total cost of the classification errors without considering test samples with
               unkown values;
            * a list of booleans indicating if the i-th test sample was classified with an unkown
              value;
            * the number of test samples classified with unkown values;
            * list where the i-th entry has the attribute index used for classification of the i-th
              sample when an unkown value occurred.

        :rtype: tuple(list[int], int, int, float, float, list[bool], int, list[int])
        """

        if self._root_node is None or self._curr_dataset is None:
            print('Decision tree must be trained before testing.')
            sys.exit(1)
        self._curr_dataset.load_test_set_from_csv(test_dataset_csv_filepath,
                                                  key_attrib_index,
                                                  class_attrib_index,
                                                  split_char,
                                                  missing_value_string)
        return self._classify_samples(self._curr_dataset.test_samples,
                                      self._curr_dataset.test_sample_class,
                                      self._curr_dataset.test_sample_costs,
                                      list(range(len(self._curr_dataset.test_sample_index_to_key))),
                                      self._curr_dataset.test_sample_index_to_key)

    def save_tree(self, filepath=None):
        """
        Saves the tree information: nodes, attributes used to split each one, values to each side,
        etc.

        :param filepath: file in which to save the tree. If `None`, prints to stdout. Defaults to
            `None`.
        :type filepath: str or None
        :return: None.
        """
        # TESTED!
        # Saves in a txt file or something similar
        def _aux_print_nominal_string(attrib_name, string_values, curr_depth):
            # TESTED!
            ret_string = '|' * curr_depth
            ret_string += '{} in {}:'.format(attrib_name, string_values)
            return ret_string

        def _aux_print_split(file_object, tree_node, curr_depth):
            # TESTED!
            if tree_node.is_leaf:
                leaf_class_string = self._curr_dataset.class_int_to_name[
                    tree_node.most_common_int_class]
                string_leaf = '|' * curr_depth + 'CLASS: ' + leaf_class_string
                print(string_leaf, file=file_object)
            else:
                attrib_index = tree_node.node_split.separation_attrib_index
                attrib_name = self._curr_dataset.attrib_names[attrib_index]
                for split_values, child_node in zip(tree_node.node_split.splits_values,
                                                    tree_node.nodes):
                    curr_string_values = sorted(
                        [self._curr_dataset.attrib_int_to_value[attrib_index][int_value]
                         for int_value in split_values])
                    print(_aux_print_nominal_string(attrib_name,
                                                    curr_string_values,
                                                    curr_depth),
                          file=file_object)
                    _aux_print_split(file_object, child_node, curr_depth + 1)

        if filepath is None:
            _aux_print_split(sys.stdout, self._root_node, curr_depth=0)
        else:
            with open(filepath, 'w') as tree_output_file:
                _aux_print_split(tree_output_file, self._root_node, curr_depth=0)


class TreeNode(object):
    """
    Contains information of a certain node of a decision tree.

    It has information about the samples used during training at this node and also about it's
    NodeSplit.

    :param dataset.Dataset curr_dataset: dataset of samples used for training/split generation.
    :param valid_samples_indices: indices of samples that should be used for training at this node.
    :type valid_samples_indices: list[int]
    :param valid_nominal_attribute: the i-th entry informs wether the i-th attribute is a valid
        nominal one.
    :type valid_samples_indices: list[bool]
    :param int max_depth_remaining: maximum depth that the subtree rooted at this node can have. If
        zero, this node will be a leaf.
    :param int min_samples_per_node: minimum number of samples that must be present in order to try
        to create a subtree rooted at this node. If less than this, this node will be a leaf.
    :param bool use_stop_conditions: informs wether we should use pruning techniques to avoid using
        attributes with small number of samples (and, thus, avoiding statistical anomalies). An
        attribute will be considered invalid if it contains less than
        `MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE` samples in the second most frequent value (this
        way at least two values have this number of samples) or if a chi-square test, applied on the
        attributes' contingency table has a p-value greater or equal to `max_p_value_chi_sq`. When
        an attribute is considered invalid for the number of samples in the second most frequent
        value, it will be considered invalid in every child node of the current TreeNode. If it was
        considered invalid because of the chi-square test, it can be considered valid in a
        descendant node. Defaults to `False`.
    :param float max_p_value_chi_sq: is the maximum p-value allowed for an attribute to be accepted
        when doing chi-square tests (that is, when `use_stop_conditions` is `True`). A p-value of
        1.0 is equal to 100%. Defaults to `0.1` (which is 10%).
    :ivar bool is_leaf: True: indicates if the current TreeNode is a tree leaf.
    :ivar int max_depth_remaining: maximum depth that the subtree rooted at the current TreeNode can
        still grow. If zero, the current TreeNode will be a leaf.
    :ivar node_split: None: Data structure containing information about which attribute and split values
        were obtained in this TreeNode with a certain criterion. Also contains the criterion value.
        It is `None` if the current TreeNode is a leaf.
    :vartype node_split: decision_tree.NodeSplit or None.
    :ivar nodes: []: list containing every child TreeNode from the current TreeNode.
    :vartype nodes: list[decision_tree.TreeNode]
    :ivar contingency_tables: contains a list where the i-th entry is a tuple containing two pieces
        of information of the i-th attribute: the contingency table for that attribute (value index
        is row, class index is column) and a list of number of times each value is attained in the
        training set (i-th entry is the number of times a sample has value i in this attribute and
        training dataset). Used by many criteria when calculating the optimal split. Note that, for
        invalid attributes, the entry is an empty decision_tree.ContingencyTable.
    :vartype contingency_tables: decision_tree.ContingencyTable or None.
    :ivar dataset.Dataset curr_dataset: dataset containing the training samples.
    :ivar valid_samples_indices: contains the indices of the valid training samples.
    :vartype valid_samples_indices: list[int]
    :ivar valid_nominal_attribute: list where the i-th entry indicates wether the i-th attribute
        from the dataset is valid and nominal or not.
    :vartype valid_nominal_attribute: list[bool]
    :vartype int num_valid_samples: number of training samples in this TreeNode.
    :vartype class_index_num_samples: list where the i-th entry indicates the number of samples
        having class i.
    :vartype class_index_num_samples: list[int]
    :vartype int most_common_int_class: index of the most frequent class.
    :vartype int number_non_empty_classes: number of classes having no sample in this TreeNode.
    """
    def __init__(self, curr_dataset, valid_samples_indices, valid_nominal_attribute,
                 max_depth_remaining, min_samples_per_node, use_stop_conditions=False,
                 max_p_value_chi_sq=0.1):
        self._use_stop_conditions = use_stop_conditions
        self._max_p_value_chi_sq = max_p_value_chi_sq

        self.max_depth_remaining = max_depth_remaining
        self._min_samples_per_node = min_samples_per_node

        self.is_leaf = True
        self.node_split = None
        self.nodes = []
        self.contingency_tables = None

        self.curr_dataset = curr_dataset
        self.valid_samples_indices = valid_samples_indices
        # Note that self.valid_nominal_attribute might be different from
        # self.curr_dataset.valid_nominal_attribute when use_stop_conditions == True.
        self.valid_nominal_attribute = valid_nominal_attribute
        self.num_valid_nominal_attributes_diff = None

        self.num_valid_samples = len(valid_samples_indices)
        self.class_index_num_samples = [0] * curr_dataset.num_classes
        # Fill self.class_index_num_samples
        for sample_index in valid_samples_indices:
            self.class_index_num_samples[
                curr_dataset.sample_class[sample_index]] += 1
        number_non_empty_classes = sum(num_samples_curr_class > 0
                                       for num_samples_curr_class in self.class_index_num_samples)
        self.most_common_int_class = self.class_index_num_samples.index(
            max(self.class_index_num_samples))

        self._calculate_contingency_tables()

    def _calculate_contingency_tables(self):
        self.contingency_tables = [] # list of `ContingencyTable`'s
        for (attrib_index,
             is_valid_nominal_attribute) in enumerate(self.valid_nominal_attribute):
            if not is_valid_nominal_attribute:
                self.contingency_tables.append(ContingencyTable())
                continue

            attrib_num_values = len(self.curr_dataset.attrib_int_to_value[attrib_index])
            curr_contingency_table = np.zeros((attrib_num_values, self.curr_dataset.num_classes),
                                              dtype=int)
            curr_values_num_samples = np.zeros((attrib_num_values), dtype=int)

            for sample_index in self.valid_samples_indices:
                curr_sample_value = self.curr_dataset.samples[sample_index][attrib_index]
                curr_sample_class = self.curr_dataset.sample_class[sample_index]
                curr_contingency_table[curr_sample_value][curr_sample_class] += 1
                curr_values_num_samples[curr_sample_value] += 1

            self.contingency_tables.append(ContingencyTable(
                contingency_table=curr_contingency_table,
                values_num_samples=curr_values_num_samples))

    def _is_attribute_valid(self, attrib_index, min_allowed_in_two_largest):
        """Returns a pair of booleans indicating:
            * wether the current attribute has more than `min_allowed_in_two_largest` samples in
              second most frequent value;
            * wether the condition above is `True` AND its chi-square test's p-value is smaller than
              `self._max_p_value_chi_sq`.
        """
        def _get_chi_square_test_p_value(contingency_table, values_num_samples):
            classes_seen = set()
            for value in range(contingency_table.shape[0]):
                for sample_class, num_samples in enumerate(contingency_table[value]):
                    if num_samples > 0 and sample_class not in classes_seen:
                        classes_seen.add(sample_class)
            num_classes = len(classes_seen)
            if num_classes == 1:
                return 0.0

            num_values = sum(num_samples > 0 for num_samples in values_num_samples)
            num_samples = sum(num_samples for num_samples in values_num_samples)
            curr_chi_square_value = 0.0
            for value, value_num_sample in enumerate(values_num_samples):
                if value_num_sample == 0:
                    continue
                for class_index in classes_seen:
                    expected_value_class = (
                        values_num_samples[value] * self.class_index_num_samples[class_index]
                        / num_samples)
                    diff = contingency_table[value][class_index] - expected_value_class
                    curr_chi_square_value += diff * (diff / expected_value_class)
            return 1. - chi2.cdf(x=curr_chi_square_value, df=((num_classes - 1) * (num_values - 1)))


        largest = 0
        second_largest = 0
        for num_samples in self.contingency_tables[attrib_index].values_num_samples:
            if num_samples > largest:
                second_largest = largest
                largest = num_samples
            elif num_samples > second_largest:
                second_largest = num_samples
        if second_largest < min_allowed_in_two_largest:
            return (False, False)
        chi_square_test_p_value = _get_chi_square_test_p_value(
            self.contingency_tables[attrib_index].contingency_table,
            self.contingency_tables[attrib_index].values_num_samples)
        return (True, chi_square_test_p_value < self._max_p_value_chi_sq)

    def create_subtree(self, criterion):
        """Given the splitting criterion, creates a tree rooted at the current TreeNode.

        :param criterion: (Criterion) splitting criterion used to create the tree recursively.
        :return: None
        """

        def _get_values_to_split(splits_values):
            values_to_split = {}
            for split_index, split_values in enumerate(splits_values):
                for value in split_values:
                    values_to_split[value] = split_index
            return values_to_split

        def _get_splits_samples_indices(num_splits, separation_attrib_index, values_to_split,
                                        valid_samples_indices, samples):
            splits_samples_indices = [[] for _ in range(num_splits)]
            for sample_index in valid_samples_indices:
                sample_value_in_split_attrib = samples[sample_index][separation_attrib_index]
                try:
                    splits_samples_indices[values_to_split[
                        sample_value_in_split_attrib]].append(sample_index)
                except KeyError:
                    print('Should not get here. Sample {} has value {} at attribute # {}, '
                          'but this value is unknown to the decision tree.'.format(
                              sample_index,
                              sample_value_in_split_attrib,
                              separation_attrib_index))
                    sys.exit(1)
            return splits_samples_indices

        def _has_multiple_nominal_values(values_num_samples):
            return sum(num_samples > 0 for num_samples in values_num_samples) > 1

        def _has_enough_samples_in_second_largest_class(class_index_num_samples,
                                                        most_common_int_class):
            second_largest = max(num_samples
                                 for class_index, num_samples in enumerate(class_index_num_samples)
                                 if class_index != most_common_int_class)
            return second_largest >= MIN_SAMPLES_SECOND_LARGEST_CLASS


        # Is it time to stop growing subtrees?
        if (self.max_depth_remaining <= 0
                or self.num_valid_samples < self._min_samples_per_node
                or self.number_non_empty_classes == 1
                or (USE_MIN_SAMPLES_SECOND_LARGEST_CLASS
                    and not _has_enough_samples_in_second_largest_class(
                        self.class_index_num_samples,
                        self.most_common_int_class))):
            return None

        # If a valid attribute has only one value, it should be marked as invalid from this node on.
        num_valid_nominal_attributes = 0
        for attrib_index, is_valid_nominal_attribute in enumerate(self.valid_nominal_attribute):
            if not is_valid_nominal_attribute:
                continue
            if (not _has_multiple_nominal_values(
                    self.contingency_tables[attrib_index].values_num_samples)):
                self.valid_nominal_attribute[attrib_index] = False
            else:
                num_valid_nominal_attributes += 1

        # If there are no valid attributes, this node should be a leaf.
        if not num_valid_nominal_attributes:
            return None

        if self._use_stop_conditions:
            num_valid_nominal_attributes = 0
            # Attributes which are valid (`True`) in `new_valid_nominal_attribute` and invalid
            # (`False`) in `new_valid_nominal_attribute_incl_chi_sq_test` should not be used to
            # split at this node, but could be used to split in descendant nodes.
            new_valid_nominal_attribute = self.valid_nominal_attribute[:]
            new_valid_nominal_attribute_incl_chi_sq_test = self.valid_nominal_attribute[:]
            for (attrib_index,
                 is_valid_nominal_attribute) in enumerate(self.valid_nominal_attribute):
                if is_valid_nominal_attribute:
                    (is_valid_num_samples,
                     is_valid_chi_sq_and_num_samples) = (self._is_attribute_valid(
                         attrib_index,
                         min_allowed_in_two_largest=MIN_SAMPLES_IN_SECOND_MOST_FREQUENT_VALUE))
                    if is_valid_chi_sq_and_num_samples:
                        num_valid_nominal_attributes += 1
                    elif is_valid_num_samples:
                        new_valid_nominal_attribute_incl_chi_sq_test[attrib_index] = False
                    else:
                        new_valid_nominal_attribute[attrib_index] = False
                        new_valid_nominal_attribute_incl_chi_sq_test[attrib_index] = False
            self.valid_nominal_attribute = new_valid_nominal_attribute_incl_chi_sq_test
            if num_valid_nominal_attributes == 0:
                return None

        # Get best split. Note that self is the current TreeNode.
        best_split = criterion.select_best_attribute_and_split(self)

        if math.isinf(best_split.criterion_value):
            # Stop condition when there is no valid attribute with more than one value (then
            # best_split.criterion_value is default, which is +- inf).
            return None

        # Calculate a list containing the inverse information of best_split.splits_values: here,
        # given a value, we want to know to which split it belongs
        values_to_split = _get_values_to_split(best_split.splits_values)

        splits_samples_indices = _get_splits_samples_indices(len(best_split.splits_values),
                                                             best_split.attrib_index,
                                                             values_to_split,
                                                             self.valid_samples_indices,
                                                             self.curr_dataset.samples)
        # Save this node's split information.
        self.node_split = NodeSplit(best_split, values_to_split)

        # Create subtrees
        self.is_leaf = False
        if self._use_stop_conditions:
            # Any attribute that has enough samples in the second most frequent value could pass the
            # chi-square test in a descendant node, thus we don't send the information of chi-square
            # test to child nodes.
            old_valid_nominal_attribute = self.valid_nominal_attribute[:]
            self.valid_nominal_attribute = new_valid_nominal_attribute
        else:
            old_valid_nominal_attribute = self.valid_nominal_attribute
        for curr_split_samples_indices in splits_samples_indices:
            self.nodes.append(TreeNode(self.curr_dataset,
                                       curr_split_samples_indices,
                                       self.valid_nominal_attribute[:],
                                       self.max_depth_remaining - 1,
                                       self._min_samples_per_node,
                                       self._use_stop_conditions,
                                       self._max_p_value_chi_sq))
            self.nodes[-1].create_subtree(criterion)
        self.valid_nominal_attribute = old_valid_nominal_attribute

    def get_most_popular_subtree(self):
        """
        :return: the number of samples in the most popular subtree. If it is leaf, returns
            `self.num_valid_samples`.
        :rtype: int
        """
        if self.is_leaf:
            return self.num_valid_samples
        return max(subtree.num_valid_samples for subtree in self.nodes)

    def prune_trivial_subtrees(self):
        """
        Applies pruning to an already trained tree. Returns the number of prunned nodes.

        If a TreeNode is trivial, that is, every leaf in its subtree has the same
        `most_common_int_class`, then the current TreeNode becomes a leaf with this class, deleting
        every child node in this process. It is applied recursively.

        :return: number of nodes prunned.
        :rtype: int
        """
        num_prunned = 0
        if not self.is_leaf:
            children_classes = set()
            num_trivial_children = 0
            for child_node in self.nodes:
                num_prunned += child_node.prune_trivial_subtrees()
                if child_node.is_leaf:
                    num_trivial_children += 1
                    children_classes.add(child_node.most_common_int_class)
            if num_trivial_children == len(self.nodes) and len(children_classes) == 1:
                self.is_leaf = True
                num_prunned += num_trivial_children
                self.nodes = []
        return num_prunned

    def get_num_nodes(self):
        """
        :return: the number of nodes in the tree rooted at the current TreeNode (counting includes
            leaves and the current TreeNode).
        :rtype: int
        """
        num_nodes = 1
        for child_node in self.nodes:
            num_nodes += child_node.get_num_nodes()
        return num_nodes

    def get_max_depth(self):
        """
        :return: the maximum depth of the tree rooted at the current TreeNode. If the current node
            is a leaf, it will return zero.
        :rtype: int
        """
        if self.is_leaf:
            return 0
        ret = 0
        for child_node in self.nodes:
            max_child_depth = child_node.get_max_depth()
            if max_child_depth > ret:
                ret = max_child_depth
        return ret + 1


class NodeSplit(object):
    """Data structure containing information about the best split found on a node.

    Used for debugging and for the classification of test samples.

    :param criteria.Split split: Split information (attribute index, split values and criterion
        value)
    :param values_to_split: reversed index for `splits_values`. Given a value, it returns the index
        of the split that this value belongs to.
    :type values_to_split: dict[int, int]
    :ival int separation_attrib_index: Index of the attribute used for splitting.
    :ival splits_values: list containing a set of attribute values for each TreeNode child. Binary
        splits have two sets (left and right split values), multiway splits may have many more.
    :valtype splits_values: list[set(int)]
    :ival values_to_split: reversed index for `splits_values`. Given a value, it returns the index
        of the split that this value belongs to.
    :valtype values_to_split: dict[int, int]
    :ival float criterion_value: criterion value for this split.
    """
    def __init__(self, split, values_to_split):
        self.separation_attrib_index = split.attrib_index
        self.splits_values = split.splits_values
        self.values_to_split = values_to_split
        self.criterion_value = split.criterion_value
