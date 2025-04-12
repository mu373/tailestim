Estimator Set
=============

.. automodule:: tailestim.estimators.estimator_set
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

.. code-block:: python

   from tailestim import TailData, TailEstimatorSet
   import matplotlib.pyplot as plt

   data = TailData(name='CAIDA_KONECT').data

   estim_set = TailEstimatorSet()
   estim_set.fit(data)

   estim_set.plot()
   plt.show()

   estim_set.plot_diagnostics()
   plt.show()