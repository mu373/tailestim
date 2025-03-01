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

    from tailestim import TailData
    from tailestim import HillEstimator
    
    data = TailData(name='Pareto').data
    
    # Initialize and fit Pickands estimator
    pickands = PickandsEstimator()
    pickands.fit(data)
    
    # Get estimated parameters
    params = pickands.get_parameters()