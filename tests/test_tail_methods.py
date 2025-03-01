import numpy as np
import pytest
from tailestim.estimators.tail_methods import (
    add_uniform_noise,
    get_distribution,
    get_ccdf
)
from tailestim.estimators.hill import HillEstimator
from tailestim.estimators.smooth_hill import SmoothHillEstimator
from tailestim.estimators.moments import MomentsEstimator
from tailestim.estimators.kernel import KernelTypeEstimator
from tailestim.estimators.pickands import PickandsEstimator

# Test data preprocessing functions
def test_add_uniform_noise():
    # Test with valid input
    data = np.array([1, 2, 3, 4, 5])
    noisy_data = add_uniform_noise(data, p=1)
    assert len(noisy_data) <= len(data)  # May be shorter due to negative value filtering
    assert np.all(noisy_data > 0)  # All values should be positive

    # Test with invalid p
    assert add_uniform_noise(data, p=0) is None

def test_get_distribution():
    data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
    x, y = get_distribution(data, number_of_bins=5)
    assert len(x) == len(y)
    assert np.all(np.array(y) >= 0)  # PDF values should be non-negative

def test_get_ccdf():
    data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
    # uniques are returned in descending order
    # ccdf is returned for each value in uniques (in descending order)
    uniques, ccdf = get_ccdf(data)
    assert len(uniques) == len(ccdf)
    assert np.all(ccdf >= 0) and np.all(ccdf <= 1)  # CCDF values should be between 0 and 1
    assert ccdf[0] == 0  # The first element of returned ccdf object is CCDF for last unique degree

# Test Hill estimator
def test_hill_estimator():
    # Generate Pareto distributed data
    np.random.seed(42)
    alpha = 2.0
    size = 1000
    data = (1/np.random.uniform(0, 1, size))**(1/alpha)
    
    # Test without bootstrap
    estimator = HillEstimator(bootstrap=False)
    estimator.fit(data)
    params = estimator.get_parameters()
    assert 'k_arr' in params
    assert 'xi_arr' in params
    assert len(params['k_arr']) == len(params['xi_arr'])
    
    # Test with bootstrap
    estimator = HillEstimator(bootstrap=True, r_bootstrap=100)
    estimator.fit(data)
    params = estimator.get_parameters()
    assert 'k_star' in params
    assert 'xi_star' in params
    assert 'gamma' in params
    assert params['gamma'] is not None

def test_smooth_hill_estimator():
    np.random.seed(42)
    data = np.random.pareto(2, 1000)
    
    estimator = SmoothHillEstimator(r_smooth=2)
    estimator.fit(data)
    params = estimator.get_parameters()
    assert 'k_arr' in params
    assert 'xi_arr' in params
    assert len(params['k_arr']) == len(params['xi_arr'])
    assert np.all(np.isfinite(params['xi_arr']))

# Test moments estimator
def test_moments_estimator():
    np.random.seed(42)
    data = np.random.pareto(2, 1000)
    
    # Test without bootstrap
    estimator = MomentsEstimator(bootstrap=False)
    estimator.fit(data)
    params = estimator.get_parameters()
    assert 'k_arr' in params
    assert 'xi_arr' in params
    assert len(params['k_arr']) == len(params['xi_arr'])
    
    # Test with bootstrap
    estimator = MomentsEstimator(bootstrap=True, r_bootstrap=100)
    estimator.fit(data)
    params = estimator.get_parameters()
    assert 'k_star' in params
    assert 'xi_star' in params
    assert params['k_star'] is not None
    assert params['xi_star'] is not None

# Test kernel estimator
def test_kernel_type_estimator():
    np.random.seed(42)
    data = np.random.pareto(2, 1000)
    
    # Test without bootstrap
    estimator = KernelTypeEstimator(hsteps=50, bootstrap=False)
    estimator.fit(data)
    params = estimator.get_parameters()
    assert 'k_arr' in params
    assert 'xi_arr' in params
    assert len(params['k_arr']) == len(params['xi_arr'])
    
    # Test with bootstrap
    estimator = KernelTypeEstimator(hsteps=50, bootstrap=True, r_bootstrap=100)
    estimator.fit(data)
    params = estimator.get_parameters()
    assert 'k_star' in params
    assert 'xi_star' in params
    assert params['k_star'] is not None
    assert params['xi_star'] is not None

# Test Pickands estimator
def test_pickands_estimator():
    np.random.seed(42)
    data = np.random.pareto(2, 1000)
    
    estimator = PickandsEstimator()
    estimator.fit(data)
    params = estimator.get_parameters()
    assert 'k_arr' in params
    assert 'xi_arr' in params
    assert len(params['k_arr']) == len(params['xi_arr'])
    assert len(params['k_arr']) <= len(data) // 4  # Pickands can only estimate up to n/4 order statistics