Smooth Hill Estimator
===================

.. currentmodule:: tailestim.estimators

.. autoclass:: tailestim.estimators.smooth_hill.SmoothHillEstimator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Examples
--------
.. code-block:: python

    from tailestim import TailData
    from tailestim import HillEstimator
    
    data = TailData(name='Pareto').data
    
    # Initialize and fit Smooth Hill estimator
    smooth_hill = SmoothHillEstimator()
    smooth_hill.fit(data)
    
    # Get estimated parameters
    params = smooth_hill.get_parameters()

