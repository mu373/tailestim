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

    from tailestim import TailData
    from tailestim import HillEstimator
    
    data = TailData(name='Pareto').data
    
    # Initialize and fit Hill estimator
    hill = HillEstimator()
    hill.fit(data)
    
    # Get estimated parameters
    params = hill.get_parameters()