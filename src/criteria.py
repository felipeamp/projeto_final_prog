#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing all criteria available for tests.


"""

import abc
import collections
import itertools
import math

import numpy as np


#: Contains the information about a given split. When empty, defaults to
#: `(None, [], float('-inf'))`.
Split = collections.namedtuple('Split',
                               ['attrib_index',
                                'splits_values',
                                'criterion_value'])
Split.__new__.__defaults__ = (None, [], float('-inf'))


class Criterion(object):
    """Abstract base class for every criterion.
    """
    __metaclass__ = abc.ABCMeta

    name = ''

    @classmethod
    @abc.abstractmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best split found, according to the criterion.
        """
        pass



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                        GINI GAIN                                          ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class GiniGain(Criterion):
    """Gini Gain criterion. For reference see "Breiman, L., Friedman, J. J., Olshen, R. A., and
    Stone, C. J. Classification and Regression Trees. Wadsworth, 1984".
    """
    name = 'Gini Gain'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """
        Returns the best attribute and its best split, according to the Gini Gain criterion.

        :param decision_tree.TreeNode tree_node: TreeNode where we want to find the best
            attribute/split.
        :return: the best Split found.
        :rtype: criteria.Split
        """
        original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
                                                  tree_node.class_index_num_samples)
        best_splits_per_attrib = []
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                splits_values = [set([value]) for value in values_seen]
                curr_children_gini_index = cls._calculate_children_gini_index(
                    tree_node.contingency_tables[attrib_index].contingency_table,
                    tree_node.contingency_tables[attrib_index].values_num_samples,
                    len(tree_node.valid_samples_indices),)
                curr_total_gini_gain = original_gini - curr_children_gini_index
                best_splits_per_attrib.append(Split(attrib_index=attrib_index,
                                                    splits_values=splits_values,
                                                    criterion_value=curr_total_gini_gain))
        if best_splits_per_attrib:
            return max(best_splits_per_attrib, key=lambda split: split.criterion_value)
        else:
            return Split()


    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @classmethod
    def _calculate_children_gini_index(cls, contingency_table, values_num_seen, num_valid_samples):
        total_children_gini = 0.0
        for value_index, value_num_samples in enumerate(values_num_seen):
            if value_num_samples == 0:
                continue
            curr_child_gini_index = cls._calculate_gini_index(value_num_samples,
                                                              contingency_table[value_index, :])
            total_children_gini += (value_num_samples / num_valid_samples) * curr_child_gini_index
        return total_children_gini

    @staticmethod
    def _calculate_gini_index(num_samples, class_num_samples):
        gini_index = 1.0
        for curr_class_num_samples in class_num_samples:
            if curr_class_num_samples > 0:
                gini_index -= (curr_class_num_samples / num_samples)**2
        return gini_index


#################################################################################################
#################################################################################################
###                                                                                           ###
###                                       TWOING                                              ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class Twoing(Criterion):
    """Twoing criterion. For reference see "Breiman, L., Friedman, J. J., Olshen, R. A., and
    Stone, C. J. Classification and Regression Trees. Wadsworth, 1984".
    """
    name = 'Twoing'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """
        Returns the best attribute and its best split, according to the Twoing criterion.

        :param decision_tree.TreeNode tree_node: TreeNode where we want to find the best
            attribute/split.
        :return: the best Split found.
        :rtype: criteria.Split
        """
        best_splits_per_attrib = []
        values_seen_per_attrib = []
        for attrib_index, is_valid_nominal_attrib in enumerate(tree_node.valid_nominal_attribute):
            if not is_valid_nominal_attrib:
                values_seen_per_attrib.append(None)
                continue
            else:
                best_total_gini_gain = float('-inf')
                best_left_values = set()
                best_right_values = set()
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                values_seen_per_attrib.append(values_seen)
                for (set_left_classes,
                     set_right_classes) in cls._generate_twoing(tree_node.class_index_num_samples):
                    (twoing_contingency_table,
                     superclass_index_num_samples) = cls._get_twoing_contingency_table(
                         tree_node.contingency_tables[attrib_index].contingency_table,
                         tree_node.contingency_tables[attrib_index].values_num_samples,
                         set_left_classes,
                         set_right_classes)
                    original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
                                                              superclass_index_num_samples)
                    (curr_gini_gain,
                     left_values,
                     right_values) = cls._two_class_trick(
                         original_gini,
                         superclass_index_num_samples,
                         values_seen,
                         tree_node.contingency_tables[attrib_index].values_num_samples,
                         twoing_contingency_table,
                         len(tree_node.valid_samples_indices))
                    if curr_gini_gain > best_total_gini_gain:
                        best_total_gini_gain = curr_gini_gain
                        best_left_values = left_values
                        best_right_values = right_values
                best_splits_per_attrib.append(Split(attrib_index=attrib_index,
                                                    splits_values=[best_left_values,
                                                                   best_right_values],
                                                    criterion_value=best_total_gini_gain))
        if best_splits_per_attrib:
            return max(best_splits_per_attrib, key=lambda split: split.criterion_value)
        else:
            return Split()

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _generate_twoing(class_index_num_samples):
        # We only need to look at superclasses of up to (len(class_index_num_samples)/2 + 1)
        # elements because of symmetry! The subsets we are not choosing are complements of the ones
        # chosen.
        non_empty_classes = set([])
        for class_index, class_num_samples in enumerate(class_index_num_samples):
            if class_num_samples > 0:
                non_empty_classes.add(class_index)
        number_non_empty_classes = len(non_empty_classes)

        for left_classes in itertools.chain.from_iterable(
                itertools.combinations(non_empty_classes, size_left_superclass)
                for size_left_superclass in range(1, number_non_empty_classes // 2 + 1)):
            set_left_classes = set(left_classes)
            set_right_classes = non_empty_classes - set_left_classes
            if not set_left_classes or not set_right_classes:
                # A valid split must have at least one sample in each side
                continue
            yield set_left_classes, set_right_classes

    @staticmethod
    def _get_twoing_contingency_table(contingency_table, values_num_samples, set_left_classes,
                                      set_right_classes):
        twoing_contingency_table = np.zeros((contingency_table.shape[0], 2), dtype=float)
        superclass_index_num_samples = [0, 0]
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            for class_index in set_left_classes:
                superclass_index_num_samples[0] += contingency_table[value][class_index]
                twoing_contingency_table[value][0] += contingency_table[value][class_index]
            for class_index in set_right_classes:
                superclass_index_num_samples[1] += contingency_table[value][class_index]
                twoing_contingency_table[value][1] += contingency_table[value][class_index]
        return twoing_contingency_table, superclass_index_num_samples

    @staticmethod
    def _two_class_trick(original_gini, class_index_num_samples, values_seen, values_num_samples,
                         contingency_table, num_total_valid_samples):
        # TESTED!
        def _get_non_empty_class_indices(class_index_num_samples):
            # TESTED!
            first_non_empty_class = None
            second_non_empty_class = None
            for class_index, class_num_samples in enumerate(class_index_num_samples):
                if class_num_samples > 0:
                    if first_non_empty_class is None:
                        first_non_empty_class = class_index
                    else:
                        second_non_empty_class = class_index
                        break
            return first_non_empty_class, second_non_empty_class

        def _calculate_value_class_ratio(values_seen, values_num_samples, contingency_table,
                                         non_empty_class_indices):
            # TESTED!
            value_number_ratio = [] # [(value, number_on_second_class, ratio_on_second_class)]
            second_class_index = non_empty_class_indices[1]
            for curr_value in values_seen:
                number_second_non_empty = contingency_table[curr_value][second_class_index]
                value_number_ratio.append((curr_value,
                                           number_second_non_empty,
                                           number_second_non_empty/values_num_samples[curr_value]))
            value_number_ratio.sort(key=lambda tup: tup[2])
            return value_number_ratio

        def _calculate_children_gini_index(num_left_first, num_left_second, num_right_first,
                                           num_right_second, num_left_samples, num_right_samples):
            # TESTED!
            if num_left_samples != 0:
                left_first_class_freq_ratio = float(num_left_first)/float(num_left_samples)
                left_second_class_freq_ratio = float(num_left_second)/float(num_left_samples)
                left_split_gini_index = (1.0
                                         - left_first_class_freq_ratio**2
                                         - left_second_class_freq_ratio**2)
            else:
                # We can set left_split_gini_index to any value here, since it will be multiplied
                # by zero in curr_children_gini_index
                left_split_gini_index = 1.0

            if num_right_samples != 0:
                right_first_class_freq_ratio = float(num_right_first)/float(num_right_samples)
                right_second_class_freq_ratio = float(num_right_second)/float(num_right_samples)
                right_split_gini_index = (1.0
                                          - right_first_class_freq_ratio**2
                                          - right_second_class_freq_ratio**2)
            else:
                # We can set right_split_gini_index to any value here, since it will be multiplied
                # by zero in curr_children_gini_index
                right_split_gini_index = 1.0

            curr_children_gini_index = ((num_left_samples * left_split_gini_index
                                         + num_right_samples * right_split_gini_index)
                                        / (num_left_samples + num_right_samples))
            return curr_children_gini_index

        # We only need to sort values by the percentage of samples in second non-empty class with
        # this value. The best split will be given by choosing an index to split this list of
        # values in two.
        (first_non_empty_class,
         second_non_empty_class) = _get_non_empty_class_indices(class_index_num_samples)
        if first_non_empty_class is None or second_non_empty_class is None:
            return (float('-inf'), {0}, set())

        value_number_ratio = _calculate_value_class_ratio(values_seen,
                                                          values_num_samples,
                                                          contingency_table,
                                                          (first_non_empty_class,
                                                           second_non_empty_class))

        best_split_total_gini_gain = float('-inf')
        best_last_left_index = 0

        num_left_first = 0
        num_left_second = 0
        num_left_samples = 0
        num_right_first = class_index_num_samples[first_non_empty_class]
        num_right_second = class_index_num_samples[second_non_empty_class]
        num_right_samples = num_total_valid_samples

        for last_left_index, (last_left_value, last_left_num_second, _) in enumerate(
                value_number_ratio[:-1]):
            num_samples_last_left_value = values_num_samples[last_left_value]
            # num_samples_last_left_value > 0 always, since the values without samples were not
            # added to the values_seen when created by cls._generate_value_to_index

            last_left_num_first = num_samples_last_left_value - last_left_num_second

            num_left_samples += num_samples_last_left_value
            num_left_first += last_left_num_first
            num_left_second += last_left_num_second
            num_right_samples -= num_samples_last_left_value
            num_right_first -= last_left_num_first
            num_right_second -= last_left_num_second

            curr_children_gini_index = _calculate_children_gini_index(num_left_first,
                                                                      num_left_second,
                                                                      num_right_first,
                                                                      num_right_second,
                                                                      num_left_samples,
                                                                      num_right_samples)
            curr_gini_gain = original_gini - curr_children_gini_index
            if curr_gini_gain > best_split_total_gini_gain:
                best_split_total_gini_gain = curr_gini_gain
                best_last_left_index = last_left_index

        # Let's get the values and split the indices corresponding to the best split found.
        set_left_values = set([tup[0] for tup in value_number_ratio[:best_last_left_index + 1]])
        set_right_values = set(values_seen) - set_left_values

        return (best_split_total_gini_gain, set_left_values, set_right_values)

    @staticmethod
    def _calculate_gini_index(side_num, class_num_side):
        gini_index = 1.0
        for curr_class_num_side in class_num_side:
            if curr_class_num_side > 0:
                gini_index -= (curr_class_num_side/side_num)**2
        return gini_index

    @classmethod
    def _calculate_children_gini_index(cls, left_num, class_num_left, right_num, class_num_right):
        left_split_gini_index = cls._calculate_gini_index(left_num, class_num_left)
        right_split_gini_index = cls._calculate_gini_index(right_num, class_num_right)
        children_gini_index = ((left_num * left_split_gini_index
                                + right_num * right_split_gini_index)
                               / (left_num + right_num))
        return children_gini_index


#################################################################################################
#################################################################################################
###                                                                                           ###
###                                       GAIN RATIO                                          ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class GainRatio(Criterion):
    """Gain Ratio criterion. For reference see "Quinlan, J. R. C4.5: Programs for Machine Learning.
    Morgan Kaufmann Publishers, 1993.".
    """
    name = 'Gain Ratio'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """
        Returns the best attribute and its best split, according to the Gain Ratio criterion.

        :param decision_tree.TreeNode tree_node: TreeNode where we want to find the best
            attribute/split.
        :return: the best Split found.
        :rtype: criteria.Split
        """
        # First we calculate the original class frequency and information
        original_information = cls._calculate_information(tree_node.class_index_num_samples,
                                                          len(tree_node.valid_samples_indices))
        best_splits_per_attrib = []
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                splits_values = [set([value]) for value in values_seen]
                curr_gain_ratio = cls._calculate_gain_ratio(
                    len(tree_node.valid_samples_indices),
                    tree_node.contingency_tables[attrib_index].contingency_table,
                    tree_node.contingency_tables[attrib_index].values_num_samples,
                    original_information)
                best_splits_per_attrib.append(Split(attrib_index=attrib_index,
                                                    splits_values=splits_values,
                                                    criterion_value=curr_gain_ratio))

        if best_splits_per_attrib:
            return max(best_splits_per_attrib, key=lambda split: split.criterion_value)
        else:
            return Split()

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @classmethod
    def _calculate_gain_ratio(cls, num_valid_samples, contingency_table, values_num_samples,
                              original_information):
        information_gain = original_information # Initial Information Gain
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            curr_split_information = cls._calculate_information(contingency_table[value],
                                                                value_num_samples)
            information_gain -= (value_num_samples / num_valid_samples) * curr_split_information

        # Gain Ratio
        potential_partition_information = cls._calculate_potential_information(values_num_samples,
                                                                               num_valid_samples)
        # Note that, since there are at least two different values, potential_partition_information
        # is never zero.
        gain_ratio = information_gain / potential_partition_information
        return gain_ratio

    @staticmethod
    def _calculate_information(class_index_num_samples, num_valid_samples):
        information = 0.0
        for curr_class_num_samples in class_index_num_samples:
            if curr_class_num_samples != 0:
                curr_frequency = curr_class_num_samples / num_valid_samples
                information -= curr_frequency * math.log2(curr_frequency)
        return information

    @staticmethod
    def _calculate_potential_information(values_num_samples, num_valid_samples):
        partition_potential_information = 0.0
        for value_num_samples in values_num_samples:
            if value_num_samples != 0:
                curr_ratio = value_num_samples / num_valid_samples
                partition_potential_information -= curr_ratio * math.log2(curr_ratio)
        return partition_potential_information


#################################################################################################
#################################################################################################
###                                                                                           ###
###                                   INFORMATION GAIN                                        ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class InformationGain(Criterion):
    """Information Gain criterion. For reference see "Quinlan, J. R. C4.5: Programs for Machine
    Learning. Morgan Kaufmann Publishers, 1993.".
    """
    name = 'Information Gain'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """
        Returns the best attribute and its best split, according to the Information Gain
        criterion.

        :param decision_tree.TreeNode tree_node: TreeNode where we want to find the best
            attribute/split.
        :return: the best Split found.
        :rtype: criteria.Split
        """
        # First we calculate the original class frequency and information
        original_information = cls._calculate_information(tree_node.class_index_num_samples,
                                                          len(tree_node.valid_samples_indices))
        best_splits_per_attrib = []
        for attrib_index, is_valid_attrib in enumerate(tree_node.valid_nominal_attribute):
            if is_valid_attrib:
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                splits_values = [set([value]) for value in values_seen]
                curr_information_gain = cls._calculate_information_gain(
                    len(tree_node.valid_samples_indices),
                    tree_node.contingency_tables[attrib_index].contingency_table,
                    tree_node.contingency_tables[attrib_index].values_num_samples,
                    original_information)
                best_splits_per_attrib.append(Split(attrib_index=attrib_index,
                                                    splits_values=splits_values,
                                                    criterion_value=curr_information_gain))

        if best_splits_per_attrib:
            return max(best_splits_per_attrib, key=lambda split: split.criterion_value)
        else:
            return Split()

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @classmethod
    def _calculate_information_gain(cls, num_valid_samples, contingency_table, values_num_samples,
                                    original_information):
        information_gain = original_information # Initial Information Gain
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            curr_split_information = cls._calculate_information(contingency_table[value],
                                                                value_num_samples)
            information_gain -= (value_num_samples / num_valid_samples) * curr_split_information
        return information_gain

    @staticmethod
    def _calculate_information(class_index_num_samples, num_valid_samples):
        information = 0.0
        for curr_class_num_samples in class_index_num_samples:
            if curr_class_num_samples != 0:
                curr_frequency = curr_class_num_samples / num_valid_samples
                information -= curr_frequency * math.log2(curr_frequency)
        return information
