Result Class
==========

.. automodule:: tailestim.estimators.result
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------
.. code-block:: python

   from tailestim import TailData
   from tailestim import HillEstimator
   
   data = TailData(name='Pareto').data
   
   # Initialize and fit Hill estimator
   hill = HillEstimator()
   hill.fit(data)
   
   # Get estimated values
   result = hill.get_result() # This returns TailEstimatorResult class.
   print(result)

   # Access individual parameters
   gamma = result.gamma_  # Power-law exponent estimate
   xi = result.xi_  # Tail index estimate
 