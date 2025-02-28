import numpy as np
import pytest
from tailestim.tail_methods import (
    add_uniform_noise,
    get_distribution,
    get_ccdf,
    get_moments_estimates_1,
    get_moments_estimates_2,
    get_moments_estimates_3,
    hill_estimator,
    smooth_hill_estimator,
    moments_estimator,
    kernel_type_estimator,
    pickands_estimator
)

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

# Test moments estimation functions
def test_get_moments_estimates():
    data = np.array([5, 4, 3, 2, 1])  # Must be in decreasing order
    
    # Test first moment
    M1 = get_moments_estimates_1(data)
    assert len(M1) == len(data) - 1
    
    # Test first and second moments
    M1, M2 = get_moments_estimates_2(data)
    assert len(M1) == len(M2) == len(data) - 1
    
    # Test all three moments
    M1, M2, M3 = get_moments_estimates_3(data)
    assert len(M1) == len(M2) == len(M3) == len(data) - 1

# Test Hill estimator
def test_hill_estimator():
    # Generate Pareto distributed data
    np.random.seed(42)
    alpha = 2.0
    size = 1000
    data = (1/np.random.uniform(0, 1, size))**(1/alpha)
    data = np.sort(data)[::-1]  # Sort in decreasing order
    
    # Test without bootstrap
    results = hill_estimator(data, bootstrap=False)
    k_arr, xi_arr = results[:2]
    assert len(k_arr) == len(xi_arr)
    
    # Test with bootstrap
    results = hill_estimator(data, bootstrap=True, r_bootstrap=100)
    k_arr, xi_arr, k_star, xi_star = results[:4]
    assert k_star is not None
    assert xi_star is not None

def test_smooth_hill_estimator():
    np.random.seed(42)
    data = np.sort(np.random.pareto(2, 1000))[::-1]
    k_arr, xi_arr = smooth_hill_estimator(data, r_smooth=2)
    assert len(k_arr) == len(xi_arr)
    assert np.all(np.isfinite(xi_arr))

# Test moments estimator
def test_moments_estimator():
    np.random.seed(42)
    data = np.sort(np.random.pareto(2, 1000))[::-1]
    
    # Test without bootstrap
    results = moments_estimator(data, bootstrap=False)
    k_arr, xi_arr = results[:2]
    assert len(k_arr) == len(xi_arr)
    
    # Test with bootstrap
    results = moments_estimator(data, bootstrap=True, r_bootstrap=100)
    k_arr, xi_arr, k_star, xi_star = results[:4]
    assert k_star is not None
    assert xi_star is not None

# Test kernel estimator
def test_kernel_type_estimator():
    np.random.seed(42)
    data = np.sort(np.random.pareto(2, 1000))[::-1]
    
    # Test without bootstrap
    results = kernel_type_estimator(data, hsteps=50, bootstrap=False)
    k_arr, xi_arr = results[:2]
    assert len(k_arr) == len(xi_arr)
    
    # Test with bootstrap
    results = kernel_type_estimator(data, hsteps=50, bootstrap=True, r_bootstrap=100)
    k_arr, xi_arr, k_star, xi_star = results[:4]
    assert k_star is not None
    assert xi_star is not None

# Test Pickands estimator
def test_pickands_estimator():
    np.random.seed(42)
    data = np.sort(np.random.pareto(2, 1000))[::-1]
    k_arr, xi_arr = pickands_estimator(data)
    assert len(k_arr) == len(xi_arr)
    assert len(k_arr) <= len(data) // 4  # Pickands can only estimate up to n/4 order statistics