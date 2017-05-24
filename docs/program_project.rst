Project
========

Architecture
############

`Object Oriented programming (OOP) <https://en.wikipedia.org/wiki/Object-oriented_programming/>`_ is a programming paradigm based on the concept of "objects", which may contain data, in the form of fields, often known as attributes; and code, in the form of procedures, often known as methods. The platform was designed using this methodology, for each requirement at least one class was conceived that addressed the issue. For instance: for requirement 1, a module containing criteria, another containing decision tree methods and another allowing to load datasets were developed. For requirement 2, each dataset has a configuration file and it is loaded through the dataset module. For requirement 3, the decision tree module has the necessary methods. The following UML diagrams show the classes developed:

.. figure:: criteria.png
    :align: center
    :figclass: align-center

    criteria UML Diagram

.. figure:: dataset.png
    :align: center
    :figclass: align-center

    dataset UML Diagram

.. figure:: decision_tree.png
    :align: center
    :figclass: align-center

    decision_tree UML Diagram

TODO: describe the classes


Testing Methodology
###################

`Test Driven Development <https://en.wikipedia.org/wiki/Test-driven_development>`_ (TDD) is a software development process that relies on the repetition of a very short development cycle: requirements are turned into very specific test cases, then the software is improved to pass the new tests, only. TDD relies in unit test, simply put a unit test is an automated code-level test for a small unit of code, like a function inside a class.

TODO: atualizar estatística
The platform was developed following a TDD. The testing coverage of platform was XX% computed using the `coverage.py <https://coverage.readthedocs.io/en/coverage-4.2/>`_.
