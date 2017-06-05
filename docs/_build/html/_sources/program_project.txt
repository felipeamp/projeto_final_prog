Project
========

Architecture
############

The platform was designed using `Object Oriented programming (OOP) <https://en.wikipedia.org/wiki/Object-oriented_programming/>`_. For each requirement at least one class (or part of it) was conceived that addressed the issue. For instance: for requirement 1, a module containing criteria, another containing decision tree methods and another allowing to load datasets were developed. For requirement 2, each dataset has a configuration file and it is loaded through the dataset module. For requirement 3, the decision tree module has the necessary methods. The following UML diagrams show the classes developed:

.. figure:: dataset.png
    :align: center
    :figclass: align-center

    dataset UML Diagram

The dataset module contains the Dataset class, which includes all the information for a given training/test dataset, and functions to load dataset configuration files.

.. figure:: criteria.png
    :align: center
    :figclass: align-center

    criteria UML Diagram

The criteria module contains different splitting criterion. Each one of them has one method, called `select_best_attribute_and_split()`. It is called in each tree node to find the best attribute to split and the best (binary or multiway) split for this attribute.

.. figure:: decision_tree.png
    :align: center
    :figclass: align-center

    decision_tree UML Diagram

The decision_tree module contains three different classes. The most basic one is a NodeSplit: it contains the splitting information of a tree node, such as the attribute to be used at the split, the values pointing to each child node, etc. The "intermediate" class is a TreeNode: it contains all the information pertained to a node in the decision tree, such as which attributes are valid, nominal or numeric, the class of each training sample that got to this node and the contingency tables of each attribute, among other things. The topmost class is a DecisionTree. It has methods to train, test, cross-validate and get tree information such as maximum depth. It also contains the root TreeNode of a tree (once trained) and the criterion to be used during training.


Testing Methodology
###################

The testing methodology used to develop the library was `Test Driven Development <https://en.wikipedia.org/wiki/Test-driven_development>`_. For that, it was used the Python UnitTesting Framework. Our library's testing coverage was of 84% computed using the `coverage.py <https://coverage.readthedocs.io/en/coverage-4.4.1/>`_ library.
