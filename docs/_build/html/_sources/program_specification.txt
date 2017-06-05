Specification
=============

Overview
########

Decision Trees are used in many applications, ranging from biomedical to credit prediction. One of the main reasons of using it as a prediction algorithm is the high interpretability of the outcome and good accuracy.

The aim of this library is to allow users to load datasets and create a decision tree to classify samples with unkown classes. It includes the main splitting criteria used in the market, such as Gini Gain, Twoing, Information Gain and Gain Ratio. A module for loading datasets and another to run experiments with decision trees are also included.

The module was develop in Python 3.6.1 using the following libraries:

    * numpy 1.12.1
    * scipy 0.19.0
    * scikit-learn 0.18.1


Requirements
############

Given the objective of the library, the following software requirements specifications were layed out:

1. Ready to Go
    * **Objective**: The library should include at least a few splitting criteria already implemented. It should also allow the user to load train and test datasets and to train/test decision trees using them.

    * **Description**: Since the main objective of the library is to let the user train decision trees on a given dataset and use its predicion on a different dataset, this is a must.

2. Reproducibility
    * **Objective**: The library should ensure reproducibility.

    * **Description**: The end user have to be able to reproduce the experiments when using the same dataset/criterion. This should be done through a dataset configuration file.

3. Performance
    * **Objective**: The library should provide mechanism for evaluating the decision trees' training and test results.

    * **Description**: The user must be able to measure the accuracy and size/depth of the trained decision trees.


Use Case
########

First the user should be able to load a train dataset. It should have a corresponding dataset configuration file.

Later, when choosing which splitting criterion to use with the given dataset, the user should be able to do a cross-validation with different criteria and choose the one that gives the best accuracy. Then, the user would train the decision tree with the chosen criterion using the whole trainind dataset.

Afterwards, the user would load a test dataset (in the same format as the training dataset) and test the accuracy of the previously trained decision tree in this test set.
