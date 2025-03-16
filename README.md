# tailestim

[GitHub](https://github.com/mu373/tailestim) | [PyPI](https://pypi.org/project/tailestim/) | [conda-forge](https://anaconda.org/conda-forge/tailestim) | [Documentation](https://tailestim.readthedocs.io/en/latest/)

[![PyPI version](https://img.shields.io/pypi/v/tailestim)](https://pypi.org/project/tailestim/) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/tailestim.svg)](https://anaconda.org/conda-forge/tailestim) [![PyPI status](https://img.shields.io/pypi/status/tailestim)](https://pypi.org/project/tailestim/)  [![Test CI status](https://github.com/mu373/tailestim/actions/workflows/test.yml/badge.svg)](https://github.com/mu373/tailestim/actions/workflows/test.yml) [![GitHub license](https://img.shields.io/github/license/mu373/tailestim)](https://github.com/mu373/tailestim/blob/main/LICENSE.txt)

A Python package for estimating tail parameters of heavy-tailed distributions, which is useful for analyzing power-law behavior in complex networks. Currently in development (alpha version).

> [!NOTE]
The original estimation implementations are from [ivanvoitalov/tail-estimation](https://github.com/ivanvoitalov/tail-estimation), which is based on the paper ["Scale-free networks well done" (Voitalov et al. 2019)](https://doi.org/10.1103/PhysRevResearch.1.033034). `tailestim` is a wrapper package that provides a more convenient/modern interface and logging, installable through `pip` and `conda`.

## Features
- Multiple estimation methods including Hill, Moments, Kernel, Pickands, and Smooth Hill estimators
- Double-bootstrap procedure for optimal threshold selection
- Built-in example datasets

## Installation
The package can be installed from [PyPI](https://pypi.org/project/tailestim/) and [conda-forge](https://anaconda.org/conda-forge/tailestim).
```bash
pip install tailestim
conda install conda-forge::tailestim
```

## Quick Start

### Using Built-in Datasets
```python
from tailestim import TailData
from tailestim import HillEstimator, KernelTypeEstimator, MomentsEstimator

# Load a sample dataset
data = TailData(name='CAIDA_KONECT').data

# Initialize and fit the Hill estimator
estimator = HillEstimator()
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
from tailestim import HillEstimator, KernelTypeEstimator, MomentsEstimator

# Create or load your network
G = nx.barabasi_albert_graph(10000, 2)
degree = list(dict(G.degree()).values()) # Degree sequence

# Initialize and fit the Hill estimator
estimator = HillEstimator()
estimator.fit(degree)

# Get the estimated parameters
result = estimator.get_parameters()
gamma = result['gamma']

# Print full results
print(estimator)
```

## Available Estimators
The package provides several estimators for tail estimation. For details on parameters that can be specified to each estimator, please refer to the original repository [ivanvoitalov/tail-estimation](https://github.com/ivanvoitalov/tail-estimation), [original paper](https://doi.org/10.1103/PhysRevResearch.1.033034), or the [actual code](https://github.com/mu373/tailestim/blob/main/src/tailestim/tail_methods.py).

1. **Hill Estimator** (`HillEstimator`)
   - Classical Hill estimator with double-bootstrap for optimal threshold selection
   - Generally recommended for power law analysis
2. **Moments Estimator** (`MomentsEstimator`)
   - Moments-based estimation with double-bootstrap
   - More robust to certain types of deviations from pure power law
3. **Kernel-type Estimator** (`KernelEstimator`)
   - Kernel-based estimation with double-bootstrap and bandwidth selection
4. **Pickands Estimator** (`PickandsEstimator`)
   - Pickands-based estimation (no bootstrap)
   - Provides arrays of estimates across different thresholds
5. **Smooth Hill Estimator** (`SmoothHillEstimator`)
   - Smoothed version of the Hill estimator (no bootstrap)

## Results
The full result can be obtained by `estimator.get_parameters()`, which returns a dictionary. This includes:
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
Tail Estimation Results (HillEstimator)
==================================================

Parameters:
--------------------
Optimal order statistic (k*): 26708
Tail index (ξ): 0.3974
Gamma (powerlaw exponent) (γ): 3.5167

Bootstrap Results:
--------------------
First bootstrap minimum AMSE fraction: 0.2744
Second bootstrap minimum AMSE fraction: 0.2745
```

## Built-in Datasets

The package includes several example datasets:
- `CAIDA_KONECT`
- `Libimseti_in_KONECT`
- `Pareto` (Follows power-law with $\gamma=2.5$)

Load any example dataset using:
```python
from tailestim import TailData
data = TailData(name='dataset_name').data
```

Loaded data 

## References
- I. Voitalov, P. van der Hoorn, R. van der Hofstad, and D. Krioukov. Scale-free networks well done. *Phys. Rev. Res.*, Oct. 2019, doi: [10.1103/PhysRevResearch.1.033034](https://doi.org/10.1103/PhysRevResearch.1.033034).
- I. Voitalov. `ivanvoitalov/tail-estimation`, GitHub. Mar. 2018. [https://github.com/ivanvoitalov/tail-estimation](https://github.com/ivanvoitalov/tail-estimation).

## Citations
If you use `tailestim` in your research or projects, I would greatly appreciate if you could cite this package, the original implementation, and the original paper (Voitalov et al. 2019).

```bibtex
@article{voitalov2019scalefree,
  title = {Scale-free networks well done},
  author = {Voitalov, Ivan and van der Hoorn, Pim and van der Hofstad, Remco and Krioukov, Dmitri},
  journal = {Phys. Rev. Res.},
  volume = {1},
  issue = {3},
  pages = {033034},
  numpages = {30},
  year = {2019},
  month = {Oct},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevResearch.1.033034},
  url = {https://link.aps.org/doi/10.1103/PhysRevResearch.1.033034}
}

@software{voitalov2018tailestimation,
  author       = {Voitalov, Ivan},
  title        = {tail-estimation},
  month        = mar,
  year         = 2018,
  publisher    = {GitHub},
  url          = {https://github.com/ivanvoitalov/tail-estimation}
}

@software{ueda2025tailestim,
  author       = {Ueda, Minami},
  title        = {tailestim: A Python package for estimating tail parameters of heavy-tailed distributions},
  month        = mar,
  year         = 2025,
  publisher    = {GitHub},
  url          = {https://github.com/mu373/tailestim}
}
```

## License
`tailestim` is distributed under the terms of the [MIT license](https://github.com/mu373/tailestim/blob/main/LICENSE.txt).
