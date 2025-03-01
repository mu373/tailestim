Moments Estimator
================

.. currentmodule:: tailestim.estimators

.. autoclass:: tailestim.estimators.moments.MomentsEstimator
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

    # Initialize and fit Moments estimator
    moments = MomentsEstimator()
    moments.fit(data)
    
    # Get estimated parameters
    params = moments.get_parameters()