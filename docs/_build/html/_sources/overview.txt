Overview
========

Detection of duplicate (DD) have been applied for several type of document collections such as: web documents, where
a DD system can assist in finding related pages, extracting structure from data an identifying web
mirrors. Another applications of DD can be found in systems like, spam detection modules, file systems,
and studies of document specific corpora.

The aim of this tool is to assists researchers and developers in determining the DD algorithm that suits
the best for his or her goals, in particular document specific corpora. The platform provides implementation of the most
successful algorithms reported in the literature i.e. MinHash [1]_ and SimHash [2]_.
Moreover the package also provides implementation of the Locality Sensitive Hashing (LSH)
indexes. A module for loading experiment configuration files, and a reporter module that applies the precision metric to the
results.

The module was develop in Python 2.7 and the requirements are the following:

    * numpy 1.11.1
    * scipy 0.18.1
    * nose  1.3.7, to execute the tests

Requirements
############

Given the objective of the library (platform) the following software requirements specifications were layout:

1. Ready to Go
    * **Objective**: The platform should include at least 2 already implemented algorithms for DD.

    * **Description**: One of the main purposes of the platform is to facilitate the exploration of DD algorithms
      so is natural that the platform provides already implements algorithms as well as a way to incorporate new
      algorithms

2. Reproducibility
    * **Objective**: The platform should ensure reproducibility.

    * **Description**: The end user have to be able to reproduce the experiments, this should be done through a configuration
      file.

3. Performance
    * **Objective**: The platform should provide mechanism for evaluating the results.

    * **Description**: The end user have to be able to measure the performance of the algorithms, so the platform should include
      already implemented.

Architecture
############

`Object Oriented programming (OOP) <https://en.wikipedia.org/wiki/Object-oriented_programming/>`_ is a programming paradigm based on
the concept of "objects", which may contain data, in the form of fields, often known as attributes; and code, in the form of procedures,
often known as methods. The platform was designed using this methodology, for each requirement a class was conceived that
addressed the issue. For instance for the requirement 2, a module that contains IO operations such reading a config file or iterating
over a directory was developed. The following UML diagram show the relationships between the developed classes, a detailed description of
each of the modules can be found in subsequent chapter.s

.. figure:: diagram.png
    :align: center
    :figclass: align-center

    Class Diagram (in UML format)

Testing Methodology
###################

`Test Driven Development <https://en.wikipedia.org/wiki/Test-driven_development>`_ (TDD) is a software development process
that relies on the repetition of a very short development cycle: requirements are turned into very specific test cases,
then the software is improved to pass the new tests, only. TDD relies in unit test, simply put a unit test is an automated
code-level test for a small unit of code, like a function inside a class.

The platform was developed following a TDD. The testing coverage of platform
was 91% computed using the `coverage.py <https://coverage.readthedocs.io/en/coverage-4.2/>`_.

.. [1] Broder, Andrei Z. "Identifying and filtering near-duplicate documents." Annual Symposium on Combinatorial Pattern Matching. Springer Berlin Heidelberg, 2000.

.. [2] Charikar, Moses S. "Similarity estimation techniques from rounding algorithms." Proceedings of the thiry-fourth annual ACM symposium on Theory of computing. ACM, 2002.
