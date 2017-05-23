User Manual
===========

This chapter illustrates how to use the platform through a practical example. The example will load documents from
a configuration file, both the true labels of the documents as well as the locations of the parent directory. Then it
will store the files. Later it will load the query documents and perform a search in the created index. The full example
can be found in the examples directory.

.. literalinclude:: ../examples/minhash_recall.py
    :linenos:
    :language: python
    :lines: 9-12

This first 4 lines import the modules to use in the rest of the script.

.. literalinclude:: ../examples/minhash_recall.py
    :linenos:
    :language: python
    :lines: 14-23

Lines 1 - 3 load the documents, first it creates an empty index (line 1), then it loads the configuration file (line 2).
Finally it reads the documents, using the documents_dir property of the config object. Lines 4 - 10 simply
insert the documents in the index.

.. literalinclude:: ../examples/minhash_recall.py
    :linenos:
    :language: python
    :lines: 25-35

As in the previous block it reads the query files and search for each of the queries in the index. The output of the
example should be::
    Approximate neighbours with Jaccard Similarity above 0.9 ['2']
    Approximate neighbours with Jaccard Similarity above 0.9 ['4']
    Approximate neighbours with Jaccard Similarity above 0.9 ['4']

In the next example we will review how to use compute the precision score using dedup.measure module

.. literalinclude:: ../examples/simhash_precision.py
    :linenos:
    :language: python

The example is basically, the same as before, but this time using SimHash, notice in line 40 a call to the precision
measure. The output of this example should be::
    Precision: 1

