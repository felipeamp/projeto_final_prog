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

dec_tree = decision_tree.DecisionTree(criteria.Twoing)
dec_tree.train(curr_dataset=train_dataset,
               training_samples_indices=list(range(train_dataset.num_samples)),
               max_depth=5,
               min_samples_per_node=1,
               use_stop_conditions=False,
               max_p_value_chi_sq=None)


test_dataset_folder = 'adult census income - test'
test_config = dataset.load_config(test_dataset_folder)
predictions = dec_tree.test_from_csv(test_dataset_csv_filepath=os.path.join(test_dataset_folder,
                                                                            'data.csv'),
                                     key_attrib_index=test_config["key attrib index"],
                                     class_attrib_index=test_config["class attrib index"],
                                     split_char=test_config["split char"],
                                     missing_value_string=test_config["missing value string"])[0]

print('\nDecision Tree predictions on test set:\n')
for test_sample_number, prediction in enumerate(predictions):
    print('\tTest sample #{}: class "{}"'.format(test_sample_number,
                                                 train_dataset.class_int_to_name[prediction]))
