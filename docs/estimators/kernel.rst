Kernel-Type Estimator
==================

.. currentmodule:: tailestim.estimators

.. autoclass:: tailestim.estimators.kernel.KernelTypeEstimator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Examples
--------
.. code-block:: python

    import numpy as np
    from tailestim.estimators import KernelTypeEstimator
    
    # Generate Pareto distributed data
    rng = np.random.default_rng(42)
    alpha = 2.0  # shape parameter
    n = 1000
    data = (1 - rng.random(n)) ** (-1/alpha)
    
    # Initialize and fit Kernel-type estimator
    kernel = KernelTypeEstimator(
        bootstrap=True,
        hsteps=200,
        alpha=0.6
    )
    kernel.fit(data)
    
    # Get estimated parameters
    params = kernel.get_parameters()