Specification
=============

Overview
########

Decision Trees are used in many applications, ranging from ... to credit prediction. One of the main reasons of using it is a high interpretability of the outcome and a good accuracy.

The aim of this library is to allow users to load datasets and create a decision tree to classify samples with unkown classes. It includes the main splitting criteria used in the market, such as Gini Gain, Twoing and Gain Ratio. A module for loading datasets and another to run experiments with decision trees are also included.

The module was develop in Python 3.6 with the following requirements:

    * numpy 1.12.1
    * scipy 0.19.0
    * scikit-learn 0.18.1

Requirements
############

Given the objective of the library, the following software requirements specifications were layout:

1. Ready to Go
    * **Objective**: The library should include at least 3 already implemented splitting criteria for decision trees. It should also allow the user to load train and tests datasets and to train/test decision trees on them.

    * **Description**: Since the main objective of the library is to let the user train decision trees on different datasets and use its predicion on a different dataset, this is a must.

2. Reproducibility
    * **Objective**: The library should ensure reproducibility.

    * **Description**: The end user have to be able to reproduce the experiments when using the same dataset/criterion. This should be done through a dataset configuration file.

3. Performance
    * **Objective**: The library should provide mechanism for evaluating the decision trees' training and test results.

    * **Description**: The user must be able to measure the accuracy and size of the decision trees.


Use Cases
#########

First the user should be able to load a train dataset. Then he should choose one of the common splitting criterion to use to train the tree (possibly through a cross-validation). Afterwards, the user should be able to test the accuracy in a validation set or in a (different) test dataset.
