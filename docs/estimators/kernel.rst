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


    from tailestim import TailData
    from tailestim import HillEstimator
    
    data = TailData(name='Pareto').data
    
    # Initialize and fit Kernel-type estimator
    kernel = KernelTypeEstimator(
        bootstrap=True,
        hsteps=200,
        alpha=0.6
    )
    kernel.fit(data)
    
    # Get estimated parameters
    params = kernel.get_parameters()