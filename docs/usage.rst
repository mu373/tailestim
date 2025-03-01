Usage Guide
==========

tailestim provides various methods for estimating tail parameters of heavy-tailed distributions, which is useful for analyzing power-law behavior in complex networks.


Installation
----------
.. code-block:: bash

   pip install tailestim


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

    # Get the estimated parameters
    result = estimator.get_parameters()
    gamma = result['gamma']

    # Print full results
    print(estimator)

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

    # Get the estimated parameters
    result = estimator.get_parameters()
    gamma = result['gamma']

    # Print full results
    print(estimator)

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

The full result can be obtained by ``estimator.get_parameters()``, which returns a dictionary. This includes:

- ``gamma``: Power law exponent (γ = 1 + 1/ξ)
- ``xi_star``: Tail index (ξ)
- ``k_star``: Optimal order statistic
- Bootstrap results (when applicable):
  - First and second bootstrap AMSE values
  - Optimal bandwidths or minimum AMSE fractions

Example Output
------------

When you ``print(estimator)`` after fitting, you will get the following output:

.. code-block:: text

    ==================================================
    Tail Estimation Results (HillEstimator)
    ==================================================

    Parameters:
    --------------------
    Optimal order statistic (k*): 26708
    Tail index (ξ): 0.3974
    Gamma (powerlaw exponent) (γ): 3.5167

    Bootstrap Results:
    --------------------
    First bootstrap minimum AMSE fraction: 0.2744
    Second bootstrap minimum AMSE fraction: 0.2745

Built-in Datasets
---------------

The package includes several example datasets:

- ``CAIDA_KONECT``
- ``Libimseti_in_KONECT``
- ``Pareto`` (Follows power-law with γ=2.5)

Load any example dataset using:

.. code-block:: python

    from tailestim import TailData
    data = TailData(name='dataset_name').data