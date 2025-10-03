import numpy as np
import pytest
pytestmark = [
    pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:divide by zero encountered in divide:RuntimeWarning")
]
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

    # Test reproducibility with same seed
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    result1 = add_uniform_noise(data, p=1, base_seed=42)
    result2 = add_uniform_noise(data, p=1, base_seed=42)
    np.testing.assert_array_equal(result1, result2)

    # Test that different seeds produce different results
    result3 = add_uniform_noise(data, p=1, base_seed=123)
    assert not np.array_equal(result1, result3)

    # Test that None seed still works (random behavior)
    result4 = add_uniform_noise(data, p=1, base_seed=None)
    assert len(result4) > 0  # Should return valid array
    assert np.all(result4 > 0)  # All values should be positive

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
    res = estimator.get_result()
    assert hasattr(res, 'k_arr_')
    assert hasattr(res, 'xi_arr_')
    assert len(res.k_arr_) == len(res.xi_arr_)

    # Test that params are returned
    params = estimator.get_params()
    assert params is not None
    
    # Test with bootstrap
    estimator = HillEstimator(bootstrap=True, r_bootstrap=100)
    estimator.fit(data)
    res = estimator.get_result()
    assert hasattr(res, 'estimator')
    assert isinstance(res.estimator, HillEstimator)
    assert hasattr(res, 'k_star_')
    assert hasattr(res, 'xi_star_')
    assert hasattr(res, 'gamma_')
    assert res.gamma_ is not None

    # Test with bootstrap with seed. Run multiple times to ensure consistency.
    bs1_results = []
    bs2_results = []
    for i in range(3):
        estimator = HillEstimator(bootstrap=True, base_seed=42, r_bootstrap=100)
        estimator.fit(data)
        res = estimator.get_result()
        bs1_kmin = res.bootstrap_results_.first_bootstrap_.k_min_
        bs2_kmin = res.bootstrap_results_.second_bootstrap_.k_min_
        bs1_results.append(bs1_kmin)
        bs2_results.append(bs2_kmin)
    # Assert that all values in bs1_results are the same
    assert all(x == bs1_results[0] for x in bs1_results), "Bootstrap results with same seed should be identical"
    assert all(x == bs2_results[0] for x in bs2_results), "Second bootstrap results with same seed should be identical"

def test_smooth_hill_estimator():
    np.random.seed(42)
    data = np.random.pareto(2, 1000)
    
    estimator = SmoothHillEstimator(r_smooth=2)
    estimator.fit(data)
    res = estimator.get_result()
    assert hasattr(res, 'estimator')
    assert isinstance(res.estimator, SmoothHillEstimator)
    assert hasattr(res, 'k_arr_')
    assert hasattr(res, 'xi_arr_')
    assert len(res.k_arr_) == len(res.xi_arr_)
    assert np.all(np.isfinite(res.xi_arr_))

    # Test that params are returned
    params = estimator.get_params()
    assert params is not None

# Test moments estimator
def test_moments_estimator():
    np.random.seed(42)
    data = np.random.pareto(2, 1000)
    
    # Test without bootstrap
    estimator = MomentsEstimator(bootstrap=False)
    estimator.fit(data)
    res = estimator.get_result()
    assert hasattr(res, 'k_arr_')
    assert hasattr(res, 'xi_arr_')
    assert len(res.k_arr_) == len(res.xi_arr_)

    # Test that params are returned
    params = estimator.get_params()
    assert params is not None
    
    # Test with bootstrap
    estimator = MomentsEstimator(bootstrap=True, r_bootstrap=100)
    estimator.fit(data)
    res = estimator.get_result()
    assert hasattr(res, 'estimator')
    assert isinstance(res.estimator, MomentsEstimator)
    assert hasattr(res, 'k_star_')
    assert hasattr(res, 'xi_star_')
    assert res.k_star_ is not None
    assert res.xi_star_ is not None

    # Test with bootstrap with seed. Run multiple times to ensure consistency.
    bs1_results = []
    bs2_results = []
    for i in range(3):
        estimator = MomentsEstimator(bootstrap=True, base_seed=42, r_bootstrap=100)
        estimator.fit(data)
        res = estimator.get_result()
        bs1_kmin = res.bootstrap_results_.first_bootstrap_.k_min_
        bs2_kmin = res.bootstrap_results_.second_bootstrap_.k_min_
        bs1_results.append(bs1_kmin)
        bs2_results.append(bs2_kmin)
    # Assert that all values in bs1_results are the same
    assert all(x == bs1_results[0] for x in bs1_results), "Bootstrap results with same seed should be identical"
    assert all(x == bs2_results[0] for x in bs2_results), "Second bootstrap results with same seed should be identical"

# Test kernel estimator
def test_kernel_type_estimator():
    np.random.seed(42)
    data = np.random.pareto(2, 1000)
    
    # Test without bootstrap
    estimator = KernelTypeEstimator(hsteps=50, bootstrap=False)
    estimator.fit(data)
    res = estimator.get_result()
    assert hasattr(res, 'k_arr_')
    assert hasattr(res, 'xi_arr_')
    assert len(res.k_arr_) == len(res.xi_arr_)

    # Test that params are returned
    params = estimator.get_params()
    assert params is not None
    
    # Test with bootstrap
    estimator = KernelTypeEstimator(hsteps=50, bootstrap=True, r_bootstrap=100)
    estimator.fit(data)
    res = estimator.get_result()
    assert hasattr(res, 'estimator')
    assert isinstance(res.estimator, KernelTypeEstimator)
    assert hasattr(res, 'k_star_')
    assert hasattr(res, 'xi_star_')
    assert res.k_star_ is not None
    assert res.xi_star_ is not None

    # Test with bootstrap with seed. Run multiple times to ensure consistency.
    bs1_results = []
    bs2_results = []
    for i in range(3):
        estimator = KernelTypeEstimator(hsteps=50, bootstrap=True, base_seed=42, r_bootstrap=100)
        estimator.fit(data)
        res = estimator.get_result()
        bs1_kmin = res.bootstrap_results_.first_bootstrap_.h_min_
        bs2_kmin = res.bootstrap_results_.second_bootstrap_.h_min_
        bs1_results.append(bs1_kmin)
        bs2_results.append(bs2_kmin)
    # Assert that all values in bs1_results are the same
    assert all(x == bs1_results[0] for x in bs1_results), "Bootstrap results with same seed should be identical"
    assert all(x == bs2_results[0] for x in bs2_results), "Second bootstrap results with same seed should be identical"

# Test Pickands estimator
def test_pickands_estimator():
    np.random.seed(42)
    data = np.random.pareto(2, 1000)
    
    estimator = PickandsEstimator()
    estimator.fit(data)
    res = estimator.get_result()
    assert hasattr(res, 'estimator')
    assert isinstance(res.estimator, PickandsEstimator)
    assert hasattr(res, 'k_arr_')
    assert hasattr(res, 'xi_arr_')
    assert len(res.k_arr_) == len(res.xi_arr_)
    assert len(res.k_arr_) <= len(data) // 4  # Pickands can only estimate up to n/4 order statistics

    # Test that params are returned
    params = estimator.get_params()
    assert params is not None