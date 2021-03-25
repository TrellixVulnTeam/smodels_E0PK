.. index:: How To's

.. |binder| image::
      https://mybinder.org/badge.svg
      :target: https://mybinder.org/v2/gh/SModelS/smodels/master?filepath=docs%2Fmanual%2Fsource%2Frecipes%2F


.. _Examples:

How To's
========



Below we provide a few examples for using SModelS and some of the :ref:`SModelS tools <smodelsTools>` as a Python library [*]_.

**To try out the examples in interactive mode:** |binder|

Main examples:
--------------

* `How To: Run SModelS using a parameter file <downloads/runWithParameterFile.html>`_ (download the Python code :download:`here <downloads/runWithParameterFile.py>`, IPython notebook :download:`here <downloads/runWithParameterFile.ipynb>`)

* `How To: Run SModelS as a python library <downloads/runAsLibrary.html>`_ (download the Python code :download:`here <downloads/runAsLibrary.py>`, IPython notebook :download:`here <downloads/runAsLibrary.ipynb>`)

.. toctree::
   :caption: Main examples:
   :titlesonly:
   :maxdepth: 1

   recipes/runWithParameterFile
   recipes/runAsLibrary
   
.. toctree::
   :caption: Examples displaying several functionalities:
   :titlesonly:
   :maxdepth: 1

   recipes/load_database
   recipes/lookup_upper_limit
   recipes/lookup_efficiency
   recipes/print_decomposition
   recipes/print_theoryPrediction
   recipes/compareUL
   recipes/lheLLPExample
   recipes/compute_likelihood
   recipes/missingTopologies
   recipes/ascii_graph_from_lhe
   recipes/marginalize

.. toctree::
   :caption: Examples using the cross-section computer:
   :titlesonly:
   :maxdepth: 1

   recipes/lo_xsecs_from_slha
   recipes/nll_xsecs_from_slha

.. toctree::
   :caption: Examples using the Database Browser:
   :titlesonly:
   :maxdepth: 1

   recipes/browserExample2
   recipes/browserExample3

.. toctree::
   :caption: Examples using the Interactive Plots tool:
   :titlesonly:
   :maxdepth: 1

   recipes/interactivePlotsExample

.. [*] Some of the output may change depending on the database version used.
