Hill Estimator
=============

.. currentmodule:: tailestim.estimators

.. autoclass:: tailestim.estimators.hill.HillEstimator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Examples
--------
.. code-block:: python

    import numpy as np
    from tailestim.estimators import HillEstimator
    
    # Generate Pareto distributed data
    rng = np.random.default_rng(42)
    alpha = 2.0  # shape parameter
    n = 1000
    data = (1 - rng.random(n)) ** (-1/alpha)
    
    # Initialize and fit Hill estimator
    hill = HillEstimator()
    hill.fit(data)
    
    # Get estimated parameters
    params = hill.get_parameters()