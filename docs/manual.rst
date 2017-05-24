User Manual
===========

This chapter illustrates how to use the library through a practical example. The example will load a training dataset using its configuration file. Then it will do a cross-validation on it and see the obtained accuracy. Four different splitting criteria will be used. Later it will train the decision tree on the whole dataset with the criterion which obtained the best accuracy. Lastly, it will test the tree on a separate set of samples.

.. literalinclude:: ../examples/cross-validate.py
    :linenos:
    :language: python
    :lines: 9-12

This first 4 lines import the modules to use in the rest of the script.

.. literalinclude:: ../examples/cross-validate.py
    :linenos:
    :language: python
    :lines: 14-23

Lines 1 - 3 load the configuration file, and then loads its associated (training) dataset. Finally it do a cross-validation with four different criteria, `Gini Gain`, `Twoing`, `Information Gain` and `Gain Ratio`.

.. literalinclude:: ../examples/cross-validate.py
    :linenos:
    :language: python
    :lines: 25-35

The output of the cross-validation, for each criterion, is:
# TODO: output

    Approximate neighbours with Jaccard Similarity above 0.9 ['2']
    Approximate neighbours with Jaccard Similarity above 0.9 ['4']
    Approximate neighbours with Jaccard Similarity above 0.9 ['4']

# TODO: criterion name
Now that we know the `` criterion had the best accuracy, we train the whole dataset with it.

.. literalinclude:: ../examples/train.py
    :linenos:
    :language: python

Lastly, we load the test dataset and classify the samples in it. Later we check with the correct classes (which are not always available) to see the accuracy.
# TODO: output
