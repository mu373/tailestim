import numpy as np
import pytest
import matplotlib.pyplot as plt
pytestmark = [
    pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:divide by zero encountered in divide:RuntimeWarning")
]
from tailestim.estimators.estimator_set import TailEstimatorSet
from tailestim.datasets import TailData

def test_tail_estimator_set_initialization():
    """Test that TailEstimatorSet can be initialized without data."""
    estimator_set = TailEstimatorSet()
    assert estimator_set.data is None
    assert estimator_set.ordered_data is None
    assert estimator_set.results is None
    assert estimator_set.fig is None
    assert estimator_set.axes is None

def test_tail_estimator_set_fit():
    """Test that TailEstimatorSet can fit data."""
    # Generate Pareto distributed data
    np.random.seed(42)
    size = 1000
    data = np.random.pareto(2, size)
    
    # Initialize
    estimator_set = TailEstimatorSet()
    
    # Fit data
    estimator_set.fit(data)
    
    # Check that data was stored and processed
    assert estimator_set.data is not None
    assert estimator_set.ordered_data is not None
    assert estimator_set.results is not None
    assert len(estimator_set.data) == size
    assert estimator_set.ordered_data[0] >= estimator_set.ordered_data[-1]  # Check ordering

def test_tail_estimator_set_plot():
    """Test that TailEstimatorSet can generate plots."""
    # Generate Pareto distributed data
    np.random.seed(42)
    data = np.random.pareto(2, 1000)
    
    estimator_set = TailEstimatorSet()
    estimator_set.fit(data)
    
    # Generate plots
    fig, axes = estimator_set.plot()
    
    # Check that plots were generated
    assert fig is not None
    assert axes is not None
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (3, 2)  # 3 rows, 2 columns of plots
    
    # Check that the figure and axes are stored in the object
    assert estimator_set.fig is fig
    assert estimator_set.axes is axes
    
    # Clean up
    plt.close(fig)

def test_tail_estimator_set_diagnostic_plot():
    """Test that TailEstimatorSet can generate diagnostic plots."""
    # Generate Pareto distributed data
    np.random.seed(42)
    data = np.random.pareto(2, 1000)
    
    # Initialize with data and enable diagnostic plots
    estimator_set = TailEstimatorSet(
        bootstrap_flag=True,
        diagnostic_plots=True,
        # r_bootstrap=100  # Reduce bootstrap iterations for faster testing
    )

    estimator_set.fit(data)
    
    # Generate diagnostic plots
    fig_d, axes_d = estimator_set.plot_diagnostics()
    
    # Check that diagnostic plots were generated
    assert fig_d is not None
    assert axes_d is not None
    assert isinstance(fig_d, plt.Figure)
    assert isinstance(axes_d, np.ndarray)
    
    # Clean up
    plt.close(fig_d)

def test_tail_estimator_set_with_built_in_dataset():
    """Test TailEstimatorSet with a built-in dataset."""
    try:
        # Load a built-in dataset
        data = TailData(name='CAIDA_KONECT').data
        
        estimator_set = TailEstimatorSet(
            bootstrap_flag=True,
            diagnostic_plots=True,
            r_bootstrap=100  # Reduce bootstrap iterations for faster testing
        )

        estimator_set.fit(data)
        
        # Generate plots
        fig, axes = estimator_set.plot()
        assert fig is not None
        assert axes is not None
        plt.close(fig)
        
        # Generate diagnostic plots
        fig_d, axes_d = estimator_set.plot_diagnostics()
        assert fig_d is not None
        assert axes_d is not None
        plt.close(fig_d)
        
    except FileNotFoundError:
        pytest.skip("Built-in dataset not found, skipping test")

def test_tail_estimator_set_errors():
    """Test that TailEstimatorSet raises appropriate errors."""
    # Initialize without data
    estimator_set = TailEstimatorSet()
    
    # Attempt to plot without fitting data
    with pytest.raises(ValueError, match="No data has been fitted"):
        estimator_set.plot()
    
    # Attempt to get diagnostic plots without fitting data
    with pytest.raises(ValueError, match="No data has been fitted"):
        estimator_set.plot_diagnostics()
    
    # Fit data but with bootstrap disabled
    np.random.seed(42)
    data = np.random.pareto(2, 1000)
    estimator_set = TailEstimatorSet(bootstrap_flag=False)
    estimator_set.fit(data)
    
    # Attempt to get diagnostic plots with bootstrap disabled
    with pytest.raises(ValueError, match="Diagnostic plots require bootstrap to be enabled"):
        estimator_set.plot_diagnostics()
    
    # Fit data with bootstrap enabled but diagnostic plots disabled
    estimator_set = TailEstimatorSet(bootstrap_flag=True, diagnostic_plots=False)
    estimator_set.fit(data)
    
    # Attempt to get diagnostic plots with diagnostic plots disabled
    with pytest.raises(ValueError, match="Diagnostic plots are not enabled"):
        estimator_set.plot_diagnostics()

def test_tail_estimator_set_parameters():
    """Test that TailEstimatorSet parameters are correctly stored and retrieved."""
    # Generate data
    np.random.seed(42)
    size = 1000
    data = np.random.pareto(2, size)
    
    # Initialize with custom parameters
    custom_bins = 50
    custom_r_smooth = 3
    custom_alpha = 0.7
    estimator_set = TailEstimatorSet(
        number_of_bins=custom_bins,
        r_smooth=custom_r_smooth,
        alpha=custom_alpha
    )

    estimator_set.fit(data)
    
    # Get parameters
    params = estimator_set.get_params()
    
    # Check that parameters were correctly stored
    assert params['number_of_bins'] == custom_bins
    assert params['r_smooth'] == custom_r_smooth
    assert params['alpha'] == custom_alpha
    assert params['data_length'] == size