User Manual
===========

This chapter illustrates how to use the library through a practical example. The example will load a training dataset using its configuration file. Then it will do a cross-validation on it and see the obtained accuracy. Four different splitting criteria will be used. Later it will train the decision tree on the whole dataset with the criterion which obtained the best accuracy. Lastly, it will test the tree on a separate set of samples.

.. literalinclude:: ../examples/example-cross-validation.py
    :linenos:
    :language: python
    :lines: 8-12

These lines import the modules used in the rest of the script.

.. literalinclude:: ../examples/example-cross-validation.py
    :linenos:
    :language: python
    :lines: 14-23

Lines 2 - 3 load the configuration file, and then lines 4-10 load its associated training dataset. We'll use the UCI's adult census income dataset: 20k samples will be used for training and the rest (little more than 12k) will be used for testing.

.. literalinclude:: ../examples/example-cross-validation.py
    :linenos:
    :language: python
    :lines: 25-47

Finally it does a cross-validation with four different criteria, `Gini Gain`, `Twoing`, `Information Gain` and `Gain Ratio`, and prints each one's accuracy.

The output of the cross-validation, for each criterion, is:

    Accuracy for criterion Gini Gain: 80.56%

    Accuracy for criterion Twoing: 84.93%

    Accuracy for criterion Information Gain: 83.58%

    Accuracy for criterion Gain Ratio: 82.76%

Now that we know the `Twoing` criterion had the best accuracy, we use the whole train dataset to train a decision tree.

.. literalinclude:: ../examples/example-train-and-test.py
    :linenos:
    :language: python
    :lines: 15-46

Lastly, we load the test dataset and classify the samples in it. Later we print the classification prediction for each sample.

    Decision Tree predictions on test set:

        Test sample #0: class "<=50K"

        Test sample #1: class ">50K"

        Test sample #2: class "<=50K"

        Test sample #3: class "<=50K"

        Test sample #4: class "<=50K"

        Test sample #5: class "<=50K"

        Test sample #6: class "<=50K"

        Test sample #7: class "<=50K"

        Test sample #8: class "<=50K"

        Test sample #9: class "<=50K"

        Test sample #10: class ">50K"

        Test sample #11: class "<=50K"

        Test sample #12: class "<=50K"

        Test sample #13: class "<=50K"

        Test sample #14: class "<=50K"

        Test sample #15: class ">50K"

        Test sample #16: class "<=50K"

        Test sample #17: class "<=50K"
    [...]
