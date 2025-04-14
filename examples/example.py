from tailestim import TailData
from tailestim import HillEstimator, KernelTypeEstimator, MomentsEstimator

from tailestim import TailEstimatorSet
import matplotlib.pyplot as plt

# Load a built-in dataset
data = TailData(name='CAIDA_KONECT').data

# Example of loading a custom dataset (commented out)
# custom_data = TailData(path='path/to/my/data.dat').data

# Initialize and fit the Hill estimator
estimator = HillEstimator(base_seed=1)
estimator.fit(data)
print(estimator)

# Get the parameters of the estimator
params = estimator.get_params()

# Get estimated values
result = estimator.get_result()

# Get the power law exponent
gamma = result.gamma_

# Print full results
print(result)

# Compare multiple estimators and plot results
estim_set = TailEstimatorSet.fit(data)
estim_set.plot()
plt.show()

# You can also plot diagnostic plots
estim_set.plot_diagnostics()
plt.show()