Usage Guide
==========

tailestim provides various methods for estimating tail parameters of heavy-tailed distributions, which is useful for analyzing power-law behavior in complex networks.


Installation
----------
This package is available from PyPI and conda-forge.

.. code-block:: bash

   pip install tailestim
   conda install conda-forge::tailestim


Quick Start
----------

Using Built-in Datasets
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from tailestim import TailData
    from tailestim import HillEstimator, KernelTypeEstimator, MomentsEstimator

    # Load a sample dataset
    data = TailData(name='CAIDA_KONECT').data

    # Initialize and fit the Hill estimator
    estimator = HillEstimator()
    estimator.fit(data)

    # Get estimated values
    result = estimator.get_result()
    gamma = result.gamma_

    # Print full results
    print(result)

Using degree sequence from networkx graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import networkx as nx
    from tailestim import HillEstimator, KernelTypeEstimator, MomentsEstimator

    # Create or load your network
    G = nx.barabasi_albert_graph(10000, 2)
    degree = list(dict(G.degree()).values()) # Degree sequence

    # Initialize and fit the Hill estimator
    estimator = HillEstimator()
    estimator.fit(degree)

    # Get estimated values
    result = estimator.get_result()
    gamma = result.gamma_

    # Print full results
    print(result)

Available Estimators
------------------

The package provides several estimators for tail estimation. For details on each estimator, refer to the respective class :doc:`API reference <api>`.

1. **Hill Estimator** (``HillEstimator``)
   - Classical Hill estimator with double-bootstrap for optimal threshold selection
   - Generally recommended for power law analysis
2. **Moments Estimator** (``MomentsEstimator``)
   - Moments-based estimation with double-bootstrap
   - More robust to certain types of deviations from pure power law
3. **Kernel-type Estimator** (``KernelEstimator``)
   - Kernel-based estimation with double-bootstrap and bandwidth selection
4. **Pickands Estimator** (``PickandsEstimator``)
   - Pickands-based estimation (no bootstrap)
   - Provides arrays of estimates across different thresholds
5. **Smooth Hill Estimator** (``SmoothHillEstimator``)
   - Smoothed version of the Hill estimator (no bootstrap)

Results
-------

The full result can be obtained by ``result = estimator.get_result()``. You can either print the result, or access individual attributes (e.g., `result.gamma_`). The output will include values such as:

- ``gamma_``: Power law exponent (γ = 1 + 1/ξ)
- ``xi_star_``: Tail index (ξ)
- ``k_star_``: Optimal order statistic
- Bootstrap results (when applicable):
  - First and second bootstrap AMSE values
  - Optimal bandwidths or minimum AMSE fractions

Example Output
------------

When you ``print(result)`` after fitting, you will get the following output:

.. code-block:: text

   --------------------------------------------------
   Result
   --------------------------------------------------
   Order statistics: Array of shape (200,) [1.0000, 1.0000, 1.0000, ...]
   Tail index estimates: Array of shape (200,) [1614487461647431761920.0000, 1249994621547387551744.0000, 967791073562264862720.0000, ...]
   Optimal order statistic (k*): 25153
   Tail index (ξ): 0.5942
   Power law exponent (γ): 2.6828
   Bootstrap Results: 
   First Bootstrap: 
      Fraction of order statistics: None
      AMSE values: None
      H Min: 0.9059
      Maximum index: None
   Second Bootstrap: 
      Fraction of order statistics: None
      AMSE values: None
      H Min: 0.9090
      Maximum index: None



Built-in Datasets and Custom Data
-------------------------------

The package includes several example datasets:

- ``CAIDA_KONECT``
- ``Libimseti_in_KONECT``
- ``Pareto`` (Follows power-law with γ=2.5)

Load any example dataset using:

.. code-block:: python

    from tailestim import TailData
    data = TailData(name='dataset_name').data

You can also load your own custom datasets by providing a path:

.. code-block:: python

    from tailestim import TailData
    data = TailData(path='path/to/my/data.dat').data

The custom data file should follow the same format as the built-in datasets:
a plain text file where each line contains two values separated by a space:
- The first value (k) is the degree or value
- The second value (n(k)) is the count or frequency of that value

For example:
```
10 3
20 2
30 1
```
This represents that there are 3 instances of value 10, 2 instances of value 20, and 1 instance of value 30.
