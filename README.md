# tailestim

A Python package for estimating tail parameters of heavy-tailed distributions, which is useful for analyzing power-law behavior in complex networks.

> [!NOTE]
> The original estimation implementations are from [ivanvoitalov/tail-estimation](https://github.com/ivanvoitalov/tail-estimation). This is a wrapper package that provides a more convenient/modern interface and logging, that can be installed using `pip` and `conda`.

## Features
- Multiple estimation methods including Hill, Moments, Kernel, Pickands, and Smooth Hill estimators
- Double-bootstrap procedure for optimal threshold selection
- Built-in dataset loader for example networks
- Support for custom network data analysis
- Comprehensive parameter estimation and diagnostics

## Installation
```bash
pip install tailestim
```

## Quick Start

### Using Built-in Datasets
```python
from tailestim.datasets import TailData
from tailestim.estimator import TailEstimator

# Load a sample dataset
data = TailData(name='CAIDA_KONECT').data

# Initialize and fit the estimator
estimator = TailEstimator(method='hill')
estimator.fit(data)

# Get the estimated parameters
result = estimator.get_parameters()
gamma = result['gamma']

# Print full results
print(estimator)
```

### Using degree sequence from networkx graphs
```python
import networkx as nx
from tailestim.estimator import TailEstimator

# Create or load your network
G = nx.barabasi_albert_graph(10000, 2)
degree = list(dict(G.degree()).values()) # Degree sequence

# Initialize and fit the estimator
estimator = TailEstimator(method='hill')
estimator.fit(degree)

# Get the estimated parameters
result = estimator.get_parameters()
gamma = result['gamma']

# Print full results
print(estimator)
```

## Available Methods
The package provides several methods for tail estimation. For details on parameters that can be specified to each methods, please refer to the original repository [ivanvoitalov/tail-estimation](https://github.com/ivanvoitalov/tail-estimation), [original paper](https://doi.org/10.1103/PhysRevResearch.1.033034), or the [actual code](https://github.com/mu373/tailestim/blob/main/src/tailestim/tail_methods.py).

1. **Hill Estimator** (`method='hill'`)
   - Classical Hill estimator with double-bootstrap for optimal threshold selection
   - Default method, generally recommended for power law analysis
2. **Moments Estimator** (`method='moments'`)
   - Moments-based estimation with double-bootstrap
   - More robust to certain types of deviations from pure power law
3. **Kernel-type Estimator** (`method='kernel'`)
   - Kernel-based estimation with double-bootstrap and bandwidth selection
   - Additional parameters: `hsteps` (int, default=200), `alpha` (float, default=0.6)
4. **Pickands Estimator** (`method='pickands'`)
   - Pickands-based estimation (no bootstrap)
   - Provides arrays of estimates across different thresholds
5. **Smooth Hill Estimator** (`method='smooth_hill'`)
   - Smoothed version of the Hill estimator (no bootstrap)
   - Additional parameter: `r_smooth` (int, default=2)

## Results
The results can be obtained by `estimator.get_parameters()`, which returns a dictionary. This includes:
- `gamma`: Power law exponent (γ = 1 + 1/ξ)
- `xi_star`: Tail index (ξ)
- `k_star`: Optimal order statistic
- Bootstrap results (when applicable):
  - First and second bootstrap AMSE values
  - Optimal bandwidths or minimum AMSE fractions

## Example Output
When you `print(estimator)` after fitting, you will get the following output.
```
==================================================
Tail Estimation Results (Hill Method)
==================================================

Parameters:
--------------------
Optimal order statistic (k*): 6873
Tail index (ξ): 0.6191
Gamma (powerlaw exponent) (γ): 2.6151

Bootstrap Results:
--------------------
First bootstrap minimum AMSE fraction: 0.6899
Second bootstrap minimum AMSE fraction: 0.6901
```

## Built-in Datasets

The package includes several example datasets:
- `CAIDA_KONECT`
- `Libimseti_in_KONECT`
- `Pareto`

Load any example dataset using:
```python
from tailestim.datasets import TailData
data = TailData(name='dataset_name').data
```

## References
- I. Voitalov, P. van der Hoorn, R. van der Hofstad, and D. Krioukov. Scale-free networks well done. *Phys. Rev. Res.*, Oct. 2019, doi: [10.1103/PhysRevResearch.1.033034](https://doi.org/10.1103/PhysRevResearch.1.033034).
- I. Voitalov. `ivanvoitalov/tail-estimation`, GitHub. Mar. 2018. [https://github.com/ivanvoitalov/tail-estimation](https://github.com/ivanvoitalov/tail-estimation).


## License
`tailestim` is distributed under the terms of the [MIT license](https://github.com/mu373/tailestim/blob/main/LICENSE.txt).
