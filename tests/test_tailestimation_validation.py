"""
Cross-package validation tests.

Compare tailestim outputs with tail-estimation to ensure
numerical equivalence when using the same random seeds.

Tests are parameterized to run across multiple datasets:
- CAIDA_KONECT: Real-world network data (26,475 points)
- Libimseti_in_KONECT: Real-world dating network data
- Pareto: Pareto-distributed data from examples (5,000 points)
- Complete graph: Synthetic complete graph degree sequence (100 points)
- Medium/Large: Synthetic Pareto distributions (1,000 and 5,000 points)
"""

import sys
import numpy as np
import pytest
from pathlib import Path
import importlib.util

# Add tail-estimation to path
TAIL_EST_PATH = Path(__file__).parent.parent / "tail-estimation" / "Python3"
TAIL_EST_FILE = TAIL_EST_PATH / "tail-estimation.py"

# Load the tail-estimation module dynamically
spec = importlib.util.spec_from_file_location("tail_estimation_old", TAIL_EST_FILE)
tail_estimation_old = importlib.util.module_from_spec(spec)
sys.modules["tail_estimation_old"] = tail_estimation_old
spec.loader.exec_module(tail_estimation_old)


# ============================================================================
# Dataset Generation and Loading
# ============================================================================


class DatasetGenerator:
    """Generate and load various datasets for testing."""

    @staticmethod
    def load_caida():
        """Load CAIDA_KONECT dataset."""
        from tailestim.datasets import TailData

        data = TailData(name="CAIDA_KONECT")
        return data.data, "CAIDA_KONECT"

    @staticmethod
    def load_libimseti():
        """Load Libimseti_in_KONECT dataset."""
        from tailestim.datasets import TailData

        data = TailData(name="Libimseti_in_KONECT")
        return data.data, "Libimseti_in_KONECT"

    @staticmethod
    def load_pareto():
        """Load Pareto dataset from examples."""
        from tailestim.datasets import TailData

        data = TailData(name="Pareto")
        return data.data, "Pareto"

    @staticmethod
    def generate_complete_graph(n_nodes=100):
        """Generate degree sequence of a complete graph."""
        # In a complete graph, each node connects to all other nodes
        # So degree of each node is (n-1)
        data = np.ones(n_nodes) * (n_nodes - 1)
        return data, f"CompleteGraph(n={n_nodes})"

    @staticmethod
    def generate_small_pareto(seed=42):
        """Generate small Pareto sample for quick tests."""
        np.random.seed(seed)
        gamma = 2.0
        alpha = gamma - 1  # Convert gamma (tail index) to numpy's alpha parameter
        data = np.random.pareto(alpha, 100) + 1
        return data, f"Small_Pareto(gamma={gamma}, n=100)"

    @staticmethod
    def generate_medium_pareto(seed=123):
        """Generate medium Pareto sample."""
        np.random.seed(seed)
        gamma = 2.0
        alpha = gamma - 1  # Convert gamma (tail index) to numpy's alpha parameter
        data = np.random.pareto(alpha, 1000) + 1
        return data, f"Medium_Pareto(gamma={gamma}, n=1000)"

    @staticmethod
    def generate_large_pareto(seed=456):
        """Generate large Pareto sample."""
        np.random.seed(seed)
        gamma = 2.0
        alpha = gamma - 1  # Convert gamma (tail index) to numpy's alpha parameter
        data = np.random.pareto(alpha, 5000) + 1
        return data, f"Large_Pareto(gamma={gamma}, n=5000)"


# Define dataset configurations for parameterized tests
DATASET_CONFIGS = [
    ("caida", DatasetGenerator.load_caida),  # CAIDA-KONECT dataset from examples
    (
        "libimseti",
        DatasetGenerator.load_libimseti,
    ),  # Libimseti_in_KONECT dataset from examples
    ("pareto", DatasetGenerator.load_pareto),  # Pareto data from examples
    ("complete_graph", lambda: DatasetGenerator.generate_complete_graph(100)),
    ("medium", DatasetGenerator.generate_medium_pareto),  # 1000 points
]

# Subset for quick tests (excluding large datasets)
QUICK_DATASETS = [
    (
        "medium",
        DatasetGenerator.generate_medium_pareto,
    ),  # Synthetic Pareto (1000 points)
    (
        "pareto",
        DatasetGenerator.load_pareto,
    ),  # Pareto data from examples (5000 points)
]

# Full test suite including large datasets
FULL_DATASETS = DATASET_CONFIGS + [
    ("large", DatasetGenerator.generate_large_pareto),
]


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(params=DATASET_CONFIGS, ids=lambda x: x[0])
def dataset(request):
    """Fixture that provides different datasets."""
    dataset_name, dataset_func = request.param
    data, description = dataset_func()
    return data, dataset_name, description


@pytest.fixture(params=QUICK_DATASETS, ids=lambda x: x[0])
def quick_dataset(request):
    """Fixture for quick tests with smaller datasets."""
    dataset_name, dataset_func = request.param
    data, description = dataset_func()
    return data, dataset_name, description


# ============================================================================
# Phase 1: Noise Function Comparison
# ============================================================================


class TestNoiseFunction:
    """Compare add_uniform_noise() implementations."""

    def test_basic_noise_equivalence(self):
        """Test that noise generation is identical with same seed."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        seed = 42

        # tail-estimation
        result_old = tail_estimation_old.add_uniform_noise(
            data.copy(), p=1, base_seed=seed
        )

        # tailestim
        from tailestim.estimators.tail_methods import add_uniform_noise as noise_new

        result_new = noise_new(data.copy(), p=1, base_seed=seed)

        # Compare
        np.testing.assert_array_almost_equal(result_old, result_new, decimal=10)

    def test_noise_different_seeds(self):
        """Test that different seeds produce different noise."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        # tail-estimation
        result_old_1 = tail_estimation_old.add_uniform_noise(
            data.copy(), p=1, base_seed=42
        )
        result_old_2 = tail_estimation_old.add_uniform_noise(
            data.copy(), p=1, base_seed=123
        )

        # tailestim
        from tailestim.estimators.tail_methods import add_uniform_noise as noise_new

        result_new_1 = noise_new(data.copy(), p=1, base_seed=42)
        result_new_2 = noise_new(data.copy(), p=1, base_seed=123)

        # Should be different across seeds
        assert not np.array_equal(result_old_1, result_old_2)
        assert not np.array_equal(result_new_1, result_new_2)

        # But same for same seed
        np.testing.assert_array_almost_equal(result_old_1, result_new_1)
        np.testing.assert_array_almost_equal(result_old_2, result_new_2)

    def test_noise_reproducibility(self):
        """Test that repeated calls with same seed give same results."""
        data = np.array([15.0, 25.0, 35.0, 45.0, 55.0])
        seed = 99

        # tailestim - multiple calls
        from tailestim.estimators.tail_methods import add_uniform_noise as noise_new

        result1 = noise_new(data.copy(), p=1, base_seed=seed)
        result2 = noise_new(data.copy(), p=1, base_seed=seed)
        result3 = noise_new(data.copy(), p=1, base_seed=seed)

        # All should be identical
        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)


# ============================================================================
# Phase 2: Hill Estimator Comparison
# ============================================================================


class TestHillEstimator:
    """Compare Hill estimator implementations across multiple datasets."""

    @pytest.mark.parametrize(
        "dataset_name,dataset_func",
        QUICK_DATASETS,
        ids=lambda x: x[0] if isinstance(x, tuple) else x,
    )
    def test_hill_basic_no_bootstrap(self, dataset_name, dataset_func):
        """Test Hill estimator without bootstrap across datasets."""
        data, description = dataset_func()

        # tail-estimation
        ordered_data = np.sort(data)[::-1]
        results_old = tail_estimation_old.hill_estimator(ordered_data, bootstrap=False)
        k_arr_old, xi_arr_old, k_star_old, xi_star_old = results_old[:4]

        # tailestim
        from tailestim.estimators import HillEstimator

        hill_new = HillEstimator(bootstrap=False)
        hill_new.fit(data)
        result_new = hill_new.get_result()

        # Compare arrays
        np.testing.assert_array_almost_equal(
            k_arr_old, result_new.k_arr_, decimal=7, err_msg=f"Failed for {description}"
        )
        np.testing.assert_array_almost_equal(
            xi_arr_old,
            result_new.xi_arr_,
            decimal=7,
            err_msg=f"Failed for {description}",
        )

    @pytest.mark.parametrize(
        "dataset_name,dataset_func",
        QUICK_DATASETS,
        ids=lambda x: x[0] if isinstance(x, tuple) else x,
    )
    def test_hill_with_bootstrap(self, dataset_name, dataset_func):
        """Test Hill estimator with bootstrap across datasets."""
        data, description = dataset_func()
        seed = 42

        # tail-estimation
        ordered_data = np.sort(data)[::-1]
        results_old = tail_estimation_old.hill_estimator(
            ordered_data,
            bootstrap=True,
            t_bootstrap=0.5,
            r_bootstrap=200,  # Reduced for speed
            eps_stop=0.99,
            verbose=False,
            diagn_plots=False,
            base_seed=seed,
        )
        k_arr_old, xi_arr_old, k_star_old, xi_star_old = results_old[:4]

        # tailestim
        from tailestim.estimators import HillEstimator

        hill_new = HillEstimator(
            bootstrap=True,
            t_bootstrap=0.5,
            r_bootstrap=200,  # Reduced for speed
            eps_stop=0.99,
            verbose=False,
            diagn_plots=False,
            base_seed=seed,
        )
        hill_new.fit(data)
        result_new = hill_new.get_result()

        # Compare core results
        np.testing.assert_array_almost_equal(
            k_arr_old,
            result_new.k_arr_,
            decimal=7,
            err_msg=f"k_arr failed for {description}",
        )
        np.testing.assert_array_almost_equal(
            xi_arr_old,
            result_new.xi_arr_,
            decimal=7,
            err_msg=f"xi_arr failed for {description}",
        )
        assert k_star_old == result_new.k_star_, (
            f"k_star mismatch for {description}: old={k_star_old}, new={result_new.k_star_}"
        )
        np.testing.assert_almost_equal(
            xi_star_old,
            result_new.xi_star_,
            decimal=7,
            err_msg=f"xi_star failed for {description}",
        )

        # Compare gamma calculation
        gamma_old = 1 + 1.0 / xi_star_old
        np.testing.assert_almost_equal(
            gamma_old,
            result_new.gamma_,
            decimal=7,
            err_msg=f"gamma failed for {description}",
        )


# ============================================================================
# Phase 3: Moments Estimator Comparison
# ============================================================================


class TestMomentsEstimator:
    """Compare Moments estimator implementations across multiple datasets."""

    @pytest.mark.parametrize(
        "dataset_name,dataset_func",
        QUICK_DATASETS,
        ids=lambda x: x[0] if isinstance(x, tuple) else x,
    )
    def test_moments_with_bootstrap(self, dataset_name, dataset_func):
        """Test Moments estimator with bootstrap across datasets."""
        data, description = dataset_func()
        seed = 99

        # tail-estimation
        ordered_data = np.sort(data)[::-1]
        results_old = tail_estimation_old.moments_estimator(
            ordered_data,
            bootstrap=True,
            t_bootstrap=0.5,
            r_bootstrap=200,  # Reduced for speed
            eps_stop=0.99,
            base_seed=seed,
        )
        k_arr_old, xi_arr_old, k_star_old, xi_star_old = results_old[:4]

        # tailestim
        from tailestim.estimators import MomentsEstimator

        moments_new = MomentsEstimator(
            bootstrap=True,
            t_bootstrap=0.5,
            r_bootstrap=200,  # Reduced for speed
            eps_stop=0.99,
            base_seed=seed,
        )
        moments_new.fit(data)
        result_new = moments_new.get_result()

        # Compare
        np.testing.assert_array_almost_equal(
            k_arr_old,
            result_new.k_arr_,
            decimal=7,
            err_msg=f"k_arr failed for {description}",
        )
        np.testing.assert_array_almost_equal(
            xi_arr_old,
            result_new.xi_arr_,
            decimal=7,
            err_msg=f"xi_arr failed for {description}",
        )
        assert k_star_old == result_new.k_star_, (
            f"k_star mismatch for {description}: old={k_star_old}, new={result_new.k_star_}"
        )
        np.testing.assert_almost_equal(
            xi_star_old,
            result_new.xi_star_,
            decimal=7,
            err_msg=f"xi_star failed for {description}",
        )


# ============================================================================
# Phase 4: Kernel Estimator Comparison
# ============================================================================


class TestKernelEstimator:
    """Compare Kernel-type estimator implementations across multiple datasets."""

    @pytest.mark.parametrize(
        "dataset_name,dataset_func",
        QUICK_DATASETS,
        ids=lambda x: x[0] if isinstance(x, tuple) else x,
    )
    def test_kernel_with_bootstrap(self, dataset_name, dataset_func):
        """Test Kernel estimator with bootstrap across datasets."""
        data, description = dataset_func()
        seed = 88
        hsteps = 50  # Smaller for speed

        # tail-estimation
        ordered_data = np.sort(data)[::-1]
        results_old = tail_estimation_old.kernel_type_estimator(
            ordered_data,
            hsteps=hsteps,
            alpha=0.6,
            bootstrap=True,
            t_bootstrap=0.5,
            r_bootstrap=200,  # Reduced for speed
            eps_stop=0.99,
            base_seed=seed,
        )
        k_arr_old, xi_arr_old, k_star_old, xi_star_old = results_old[:4]

        # tailestim
        from tailestim.estimators import KernelTypeEstimator

        kernel_new = KernelTypeEstimator(
            hsteps=hsteps,
            alpha=0.6,
            bootstrap=True,
            t_bootstrap=0.5,
            r_bootstrap=200,  # Reduced for speed
            eps_stop=0.99,
            base_seed=seed,
        )
        kernel_new.fit(data)
        result_new = kernel_new.get_result()

        # Compare
        np.testing.assert_array_almost_equal(
            k_arr_old,
            result_new.k_arr_,
            decimal=7,
            err_msg=f"k_arr failed for {description}",
        )
        np.testing.assert_array_almost_equal(
            xi_arr_old,
            result_new.xi_arr_,
            decimal=7,
            err_msg=f"xi_arr failed for {description}",
        )
        assert k_star_old == result_new.k_star_, (
            f"k_star mismatch for {description}: old={k_star_old}, new={result_new.k_star_}"
        )
        np.testing.assert_almost_equal(
            xi_star_old,
            result_new.xi_star_,
            decimal=7,
            err_msg=f"xi_star failed for {description}",
        )


# ============================================================================
# Phase 5: Pickands Estimator Comparison
# ============================================================================


class TestPickandsEstimator:
    """Compare Pickands estimator implementations across multiple datasets."""

    def test_pickands_equivalence(self, dataset):
        """Test Pickands estimator (no bootstrap) across datasets."""
        data, dataset_name, description = dataset

        # tail-estimation
        ordered_data = np.sort(data)[::-1]
        k_arr_old, xi_arr_old = tail_estimation_old.pickands_estimator(ordered_data)

        # tailestim
        from tailestim.estimators import PickandsEstimator

        pickands_new = PickandsEstimator()
        pickands_new.fit(data)
        result_new = pickands_new.get_result()

        # Compare
        np.testing.assert_array_almost_equal(
            k_arr_old,
            result_new.k_arr_,
            decimal=7,
            err_msg=f"k_arr failed for {description}",
        )
        np.testing.assert_array_almost_equal(
            xi_arr_old,
            result_new.xi_arr_,
            decimal=7,
            err_msg=f"xi_arr failed for {description}",
        )


# ============================================================================
# Phase 6: Comprehensive Multi-Dataset Validation
# ============================================================================


class TestComprehensiveValidation:
    """Comprehensive tests across all datasets and estimators."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "dataset_name,dataset_func",
        FULL_DATASETS,
        ids=lambda x: x[0] if isinstance(x, tuple) else x,
    )
    def test_all_estimators_on_dataset(self, dataset_name, dataset_func):
        """Run all estimators on each dataset with same seed."""
        data, description = dataset_func()
        seed = 42

        print(f"\n{'=' * 60}")
        print(f"Testing dataset: {description}")
        print(f"Data size: {len(data)}")
        print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"{'=' * 60}")

        # Test Hill
        from tailestim.estimators import HillEstimator

        hill = HillEstimator(bootstrap=True, r_bootstrap=100, base_seed=seed)
        hill.fit(data)
        hill_result = hill.get_result()
        print(f"Hill: γ = {hill_result.gamma_:.4f}, k* = {hill_result.k_star_}")

        # Test Moments
        from tailestim.estimators import MomentsEstimator

        moments = MomentsEstimator(bootstrap=True, r_bootstrap=100, base_seed=seed)
        moments.fit(data)
        moments_result = moments.get_result()
        print(
            f"Moments: γ = {moments_result.gamma_:.4f}, k* = {moments_result.k_star_}"
        )

        # Test Kernel (may fail for degenerate data like complete graphs)
        from tailestim.estimators import KernelTypeEstimator

        kernel = KernelTypeEstimator(
            bootstrap=True, r_bootstrap=100, hsteps=50, base_seed=seed
        )
        try:
            kernel.fit(data)
            kernel_result = kernel.get_result()
            print(
                f"Kernel: γ = {kernel_result.gamma_:.4f}, k* = {kernel_result.k_star_}"
            )
            kernel_succeeded = True
        except ValueError as e:
            # Kernel can fail on degenerate data (e.g., all values identical)
            print(f"Kernel: Failed ({str(e)})")
            kernel_succeeded = False

        # Test Pickands
        from tailestim.estimators import PickandsEstimator

        pickands = PickandsEstimator()
        pickands.fit(data)
        pickands_result = pickands.get_result()
        print("Pickands: arrays generated")

        # All should have completed without errors
        assert hill_result.k_star_ is not None
        assert moments_result.k_star_ is not None
        if kernel_succeeded:
            assert kernel_result.k_star_ is not None
        assert len(pickands_result.k_arr_) > 0


# ============================================================================
# Phase 7: Edge Cases and Reproducibility
# ============================================================================


class TestReproducibility:
    """Test reproducibility across runs and seeds."""

    def test_multiple_seeds_consistency(self):
        """Test that various seeds all produce consistent results."""
        data, _ = DatasetGenerator.generate_small_pareto()
        seeds = [1, 42, 100, 999, 12345]

        for seed in seeds:
            # tail-estimation
            ordered_data = np.sort(data)[::-1]
            results_old = tail_estimation_old.hill_estimator(
                ordered_data, bootstrap=True, r_bootstrap=100, base_seed=seed
            )
            k_star_old, xi_star_old = results_old[2:4]

            # tailestim
            from tailestim.estimators import HillEstimator

            hill_new = HillEstimator(bootstrap=True, r_bootstrap=100, base_seed=seed)
            hill_new.fit(data)
            result_new = hill_new.get_result()

            # Compare
            assert k_star_old == result_new.k_star_, f"Seed {seed}: k_star mismatch"
            np.testing.assert_almost_equal(
                xi_star_old,
                result_new.xi_star_,
                decimal=7,
                err_msg=f"Seed {seed}: xi_star mismatch",
            )

    def test_reproducibility_across_runs(self):
        """Test that results are reproducible across multiple runs."""
        data, _ = DatasetGenerator.generate_small_pareto()
        seed = 42

        # Run tailestim multiple times
        from tailestim.estimators import HillEstimator

        results = []
        for _ in range(3):
            hill = HillEstimator(bootstrap=True, r_bootstrap=100, base_seed=seed)
            hill.fit(data)
            result = hill.get_result()
            results.append((result.k_star_, result.xi_star_, result.gamma_))

        # All runs should produce identical results
        for i in range(1, len(results)):
            assert results[0][0] == results[i][0]  # k_star
            np.testing.assert_almost_equal(results[0][1], results[i][1])  # xi_star
            np.testing.assert_almost_equal(results[0][2], results[i][2])  # gamma


# ============================================================================
# Phase 8: Plot Data Comparison
# ============================================================================


class TestPlotDataComparison:
    """Compare plotting data between tail-estimation and tailestim to ensure identical plots."""

    @pytest.mark.parametrize(
        "dataset_name,dataset_func",
        QUICK_DATASETS,
        ids=lambda x: x[0] if isinstance(x, tuple) else x,
    )
    def test_pdf_ccdf_data(self, dataset_name, dataset_func):
        """Test that PDF and CCDF plotting data is identical."""
        data, description = dataset_func()

        # tail-estimation
        ordered_data = np.sort(data)[::-1]
        x_pdf_old, y_pdf_old = tail_estimation_old.get_distribution(
            ordered_data, number_of_bins=30
        )
        x_ccdf_old, y_ccdf_old = tail_estimation_old.get_ccdf(ordered_data)

        # tailestim
        from tailestim.estimators.tail_methods import get_distribution, get_ccdf

        x_pdf_new, y_pdf_new = get_distribution(ordered_data, number_of_bins=30)
        x_ccdf_new, y_ccdf_new = get_ccdf(ordered_data)

        # Compare PDF data
        np.testing.assert_array_almost_equal(
            x_pdf_old,
            x_pdf_new,
            decimal=10,
            err_msg=f"PDF x values differ for {description}",
        )
        np.testing.assert_array_almost_equal(
            y_pdf_old,
            y_pdf_new,
            decimal=10,
            err_msg=f"PDF y values differ for {description}",
        )

        # Compare CCDF data
        np.testing.assert_array_almost_equal(
            x_ccdf_old,
            x_ccdf_new,
            decimal=10,
            err_msg=f"CCDF x values differ for {description}",
        )
        np.testing.assert_array_almost_equal(
            y_ccdf_old,
            y_ccdf_new,
            decimal=10,
            err_msg=f"CCDF y values differ for {description}",
        )

    @pytest.mark.parametrize(
        "dataset_name,dataset_func",
        QUICK_DATASETS,
        ids=lambda x: x[0] if isinstance(x, tuple) else x,
    )
    def test_hill_bootstrap_plot_data(self, dataset_name, dataset_func):
        """Test that Hill estimator bootstrap AMSE plot data is identical."""
        data, description = dataset_func()
        seed = 42

        # tail-estimation
        ordered_data = np.sort(data)[::-1]
        results_old = tail_estimation_old.hill_estimator(
            ordered_data,
            bootstrap=True,
            r_bootstrap=200,
            base_seed=seed,
            diagn_plots=True,
        )
        # Extract bootstrap results (indices 4-7)
        if len(results_old) > 4:
            x1_old, n1_amse_old, k1_old, max_index1_old = results_old[4:8]
            x2_old, n2_amse_old, k2_old, max_index2_old = results_old[8:12]

            # tailestim
            from tailestim.estimators import HillEstimator

            hill_new = HillEstimator(
                bootstrap=True, r_bootstrap=200, base_seed=seed, diagn_plots=True
            )
            hill_new.fit(data)
            result_new = hill_new.get_result()

            # Extract bootstrap results
            x1_new = result_new.bootstrap_results_.first_bootstrap_.x_arr_
            n1_amse_new = result_new.bootstrap_results_.first_bootstrap_.amse_
            k1_new = result_new.bootstrap_results_.first_bootstrap_.k_min_
            max_index1_new = result_new.bootstrap_results_.first_bootstrap_.max_index_

            x2_new = result_new.bootstrap_results_.second_bootstrap_.x_arr_
            n2_amse_new = result_new.bootstrap_results_.second_bootstrap_.amse_
            k2_new = result_new.bootstrap_results_.second_bootstrap_.k_min_
            max_index2_new = result_new.bootstrap_results_.second_bootstrap_.max_index_

            # Compare first bootstrap
            np.testing.assert_array_almost_equal(
                x1_old,
                x1_new,
                decimal=10,
                err_msg=f"First bootstrap x_arr differs for {description}",
            )
            np.testing.assert_array_almost_equal(
                n1_amse_old,
                n1_amse_new,
                decimal=10,
                err_msg=f"First bootstrap AMSE differs for {description}",
            )
            np.testing.assert_almost_equal(
                k1_old,
                k1_new,
                decimal=10,
                err_msg=f"First bootstrap k_min differs for {description}",
            )
            assert max_index1_old == max_index1_new, (
                f"First bootstrap max_index differs for {description}"
            )

            # Compare second bootstrap
            np.testing.assert_array_almost_equal(
                x2_old,
                x2_new,
                decimal=10,
                err_msg=f"Second bootstrap x_arr differs for {description}",
            )
            np.testing.assert_array_almost_equal(
                n2_amse_old,
                n2_amse_new,
                decimal=10,
                err_msg=f"Second bootstrap AMSE differs for {description}",
            )
            np.testing.assert_almost_equal(
                k2_old,
                k2_new,
                decimal=10,
                err_msg=f"Second bootstrap k_min differs for {description}",
            )
            assert max_index2_old == max_index2_new, (
                f"Second bootstrap max_index differs for {description}"
            )

    @pytest.mark.parametrize(
        "dataset_name,dataset_func",
        QUICK_DATASETS,
        ids=lambda x: x[0] if isinstance(x, tuple) else x,
    )
    def test_moments_bootstrap_plot_data(self, dataset_name, dataset_func):
        """Test that Moments estimator bootstrap AMSE plot data is identical."""
        data, description = dataset_func()
        seed = 99

        # tail-estimation
        ordered_data = np.sort(data)[::-1]
        results_old = tail_estimation_old.moments_estimator(
            ordered_data,
            bootstrap=True,
            r_bootstrap=200,
            base_seed=seed,
            diagn_plots=True,
        )
        # Extract bootstrap results
        if len(results_old) > 4:
            x1_old, n1_amse_old, k1_old, max_index1_old = results_old[4:8]
            x2_old, n2_amse_old, k2_old, max_index2_old = results_old[8:12]

            # tailestim
            from tailestim.estimators import MomentsEstimator

            moments_new = MomentsEstimator(
                bootstrap=True, r_bootstrap=200, base_seed=seed, diagn_plots=True
            )
            moments_new.fit(data)
            result_new = moments_new.get_result()

            # Extract bootstrap results
            x1_new = result_new.bootstrap_results_.first_bootstrap_.x_arr_
            n1_amse_new = result_new.bootstrap_results_.first_bootstrap_.amse_
            k1_new = result_new.bootstrap_results_.first_bootstrap_.k_min_
            max_index1_new = result_new.bootstrap_results_.first_bootstrap_.max_index_

            x2_new = result_new.bootstrap_results_.second_bootstrap_.x_arr_
            n2_amse_new = result_new.bootstrap_results_.second_bootstrap_.amse_
            k2_new = result_new.bootstrap_results_.second_bootstrap_.k_min_
            max_index2_new = result_new.bootstrap_results_.second_bootstrap_.max_index_

            # Compare first bootstrap
            np.testing.assert_array_almost_equal(
                x1_old,
                x1_new,
                decimal=10,
                err_msg=f"First bootstrap x_arr differs for {description}",
            )
            np.testing.assert_array_almost_equal(
                n1_amse_old,
                n1_amse_new,
                decimal=10,
                err_msg=f"First bootstrap AMSE differs for {description}",
            )
            np.testing.assert_almost_equal(
                k1_old,
                k1_new,
                decimal=10,
                err_msg=f"First bootstrap k_min differs for {description}",
            )
            assert max_index1_old == max_index1_new, (
                f"First bootstrap max_index differs for {description}"
            )

            # Compare second bootstrap
            np.testing.assert_array_almost_equal(
                x2_old,
                x2_new,
                decimal=10,
                err_msg=f"Second bootstrap x_arr differs for {description}",
            )
            np.testing.assert_array_almost_equal(
                n2_amse_old,
                n2_amse_new,
                decimal=10,
                err_msg=f"Second bootstrap AMSE differs for {description}",
            )
            np.testing.assert_almost_equal(
                k2_old,
                k2_new,
                decimal=10,
                err_msg=f"Second bootstrap k_min differs for {description}",
            )
            assert max_index2_old == max_index2_new, (
                f"Second bootstrap max_index differs for {description}"
            )

    @pytest.mark.parametrize(
        "dataset_name,dataset_func",
        QUICK_DATASETS,
        ids=lambda x: x[0] if isinstance(x, tuple) else x,
    )
    def test_kernel_bootstrap_plot_data(self, dataset_name, dataset_func):
        """Test that Kernel estimator bootstrap AMSE plot data is identical."""
        data, description = dataset_func()
        seed = 88
        hsteps = 50

        # tail-estimation
        ordered_data = np.sort(data)[::-1]
        try:
            results_old = tail_estimation_old.kernel_type_estimator(
                ordered_data,
                hsteps=hsteps,
                bootstrap=True,
                r_bootstrap=200,
                base_seed=seed,
                diagn_plots=True,
            )
        except ValueError:
            # If old package fails, new package should also fail
            from tailestim.estimators import KernelTypeEstimator

            with pytest.raises(ValueError):
                kernel_new = KernelTypeEstimator(
                    hsteps=hsteps,
                    bootstrap=True,
                    r_bootstrap=200,
                    base_seed=seed,
                    diagn_plots=True,
                )
                kernel_new.fit(data)
            return  # Both failed as expected, test passes

        # Extract bootstrap results
        if len(results_old) > 4:
            x1_old, n1_amse_old, h1_old, max_index1_old = results_old[4:8]
            x2_old, n2_amse_old, h2_old, max_index2_old = results_old[8:12]

            # tailestim
            from tailestim.estimators import KernelTypeEstimator

            kernel_new = KernelTypeEstimator(
                hsteps=hsteps,
                bootstrap=True,
                r_bootstrap=200,
                base_seed=seed,
                diagn_plots=True,
            )
            kernel_new.fit(data)
            result_new = kernel_new.get_result()

            # Extract bootstrap results
            x1_new = result_new.bootstrap_results_.first_bootstrap_.x_arr_
            n1_amse_new = result_new.bootstrap_results_.first_bootstrap_.amse_
            h1_new = result_new.bootstrap_results_.first_bootstrap_.h_min_
            max_index1_new = result_new.bootstrap_results_.first_bootstrap_.max_index_

            x2_new = result_new.bootstrap_results_.second_bootstrap_.x_arr_
            n2_amse_new = result_new.bootstrap_results_.second_bootstrap_.amse_
            h2_new = result_new.bootstrap_results_.second_bootstrap_.h_min_
            max_index2_new = result_new.bootstrap_results_.second_bootstrap_.max_index_

            # Compare first bootstrap
            np.testing.assert_array_almost_equal(
                x1_old,
                x1_new,
                decimal=10,
                err_msg=f"First bootstrap x_arr differs for {description}",
            )
            np.testing.assert_array_almost_equal(
                n1_amse_old,
                n1_amse_new,
                decimal=10,
                err_msg=f"First bootstrap AMSE differs for {description}",
            )
            np.testing.assert_almost_equal(
                h1_old,
                h1_new,
                decimal=10,
                err_msg=f"First bootstrap h_min differs for {description}",
            )
            np.testing.assert_almost_equal(
                max_index1_old,
                max_index1_new,
                decimal=10,
                err_msg=f"First bootstrap max_index differs for {description}",
            )

            # Compare second bootstrap
            np.testing.assert_array_almost_equal(
                x2_old,
                x2_new,
                decimal=10,
                err_msg=f"Second bootstrap x_arr differs for {description}",
            )
            np.testing.assert_array_almost_equal(
                n2_amse_old,
                n2_amse_new,
                decimal=10,
                err_msg=f"Second bootstrap AMSE differs for {description}",
            )
            np.testing.assert_almost_equal(
                h2_old,
                h2_new,
                decimal=10,
                err_msg=f"Second bootstrap k_min differs for {description}",
            )
            np.testing.assert_almost_equal(
                max_index2_old,
                max_index2_new,
                decimal=10,
                err_msg=f"Second bootstrap max_index differs for {description}",
            )
