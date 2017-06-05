#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright Felipe de Albuquerque Mello Pereira 2017

import sys
sys.path.insert(0, '../src')

import os

import criteria
import dataset
import decision_tree

# dataset.load_config takes the path to the dataset folder.
train_dataset_folder = 'adult census income - train'
train_config = dataset.load_config(train_dataset_folder)
train_dataset = dataset.Dataset(training_dataset_csv_filepath=os.path.join(train_dataset_folder,
                                                                           'data.csv'),
                                key_attrib_index=train_config["key attrib index"],
                                class_attrib_index=train_config["class attrib index"],
                                split_char=train_config["split char"],
                                missing_value_string=train_config["missing value string"],
                                load_numeric=True)

criteria_list = [criteria.GiniGain,
                 criteria.Twoing,
                 criteria.InformationGain,
                 criteria.GainRatio]
results = []
for criterion in criteria_list:
    dec_tree = decision_tree.DecisionTree(criterion)
    num_correct_classifications = dec_tree.cross_validate(curr_dataset=train_dataset,
                                                          num_folds=3,
                                                          max_depth=5,
                                                          min_samples_per_node=1,
                                                          is_stratified=True,
                                                          print_tree=False,
                                                          seed=1,
                                                          print_samples=False,
                                                          use_stop_conditions=False,
                                                          max_p_value_chi_sq=None)[1]
    accuracy = 100.0 * num_correct_classifications / train_dataset.num_samples
    results.append((criterion.name, accuracy))

print()
for result in results:
    print('Accuracy for criterion {}: {:.2f}%'.format(result[0], result[1]))
