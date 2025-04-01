from tailestim import TailData
from tailestim import HillEstimator, KernelTypeEstimator, MomentsEstimator

# Load a sample dataset
data = TailData(name='CAIDA_KONECT').data

# Initialize and fit the Hill estimator
estimator = HillEstimator(base_seed=1)
estimator.fit(data)
print(estimator)

# Get the estimated parameters
result = estimator.get_parameters()

# Get the power law exponent
gamma = result['gamma']

# Print full results
print(result)