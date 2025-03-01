Pickands Estimator
================

.. currentmodule:: tailestim.estimators

.. autoclass:: tailestim.estimators.pickands.PickandsEstimator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Examples
--------
.. code-block:: python

    import numpy as np
    from tailestim.estimators import PickandsEstimator
    
    # Generate Pareto distributed data
    rng = np.random.default_rng(42)
    alpha = 2.0  # shape parameter
    n = 1000
    data = (1 - rng.random(n)) ** (-1/alpha)
    
    # Initialize and fit Pickands estimator
    pickands = PickandsEstimator()
    pickands.fit(data)
    
    # Get estimated parameters
    params = pickands.get_parameters()