import numpy as np
from matplotlib import pyplot as plt
from typing import Optional, Tuple, Union, Dict, Any, List
from numpy.random import BitGenerator, SeedSequence, RandomState, Generator
from .bulk_fit import fit_estimators
from .plot.plot_methods import make_plots

class TailEstimatorSet:
    """
    Class for running estimation with multiple estimator methods at once and creating a plot for comparison.
    
    Parameters
    ----------
    data : np.ndarray
        The data to plot.
    output_file_path : str, optional
        File path to which plots should be saved. If None, the figure is not saved.
    number_of_bins : int, default=30
        Number of log-bins for degree distribution.
    r_smooth : int, default=2
        Integer parameter controlling the width of smoothing window.
        Typically small value such as 2 or 3.
    alpha : float, default=0.6
        Parameter controlling the amount of "smoothing" for the kernel-type estimator.
        Should be greater than 0.5.
    hsteps : int, default=200
        Parameter controlling number of bandwidth steps of the kernel-type estimator.
    bootstrap_flag : bool, default=True
        Flag to switch on/off double-bootstrap procedure.
    t_bootstrap : float, default=0.5
        Parameter controlling the size of the 2nd bootstrap.
        Defined from n2 = n*(t_bootstrap).
    r_bootstrap : int, default=500
        Number of bootstrap resamplings for the 1st and 2nd bootstraps.
    diagnostic_plots : bool, default=False
        Flag to switch on/off generation of AMSE diagnostic plots.
    eps_stop : float, default=1.0
        Parameter controlling range of AMSE minimization.
        Defined as the fraction of order statistics to consider
        during the AMSE minimization step.
    theta1 : float, default=0.01
        Lower bound of plotting range, defined as k_min = ceil(n^theta1).
        Overwritten if plots behave badly within the range.
    theta2 : float, default=0.99
        Upper bound of plotting range, defined as k_max = floor(n^theta2).
        Overwritten if plots behave badly within the range.
    verbose : bool, default=False
        Flag controlling bootstrap verbosity.
    noise_flag : bool, default=True
        Switch on/off uniform noise in range [-5*10^(-p), 5*10^(-p)]
        that is added to each data point. Used for integer-valued sequences.
    p_noise : int, default=1
        Integer parameter controlling noise amplitude.
    savedata : bool, default=False
        Flag to save data files in the directory with plots.
    auto_plot : bool, default=False
        Whether to create the plots immediately upon initialization.
    base_seed: None | SeedSequence | BitGenerator | Generator | RandomState, default=None
        Base random seed for reproducibility of bootstrap. Only used for methods with bootstrap.
    

    """
    
    def __init__(
        self,
        data: np.ndarray = None,
        output_file_path: Optional[str] = None,
        number_of_bins: int = 30,
        r_smooth: int = 2,
        alpha: float = 0.6,
        hsteps: int = 200,
        bootstrap_flag: bool = True,
        t_bootstrap: float = 0.5,
        r_bootstrap: int = 500,
        diagnostic_plots: bool = False,
        eps_stop: float = 1.0,
        theta1: float = 0.01,
        theta2: float = 0.99,
        verbose: bool = False,
        noise_flag: bool = True,
        p_noise: int = 1,
        savedata: bool = False,
        auto_plot: bool = False,
        base_seed: Union[None, SeedSequence, BitGenerator, Generator, RandomState] = None
    ):
        # Store parameters
        self.output_file_path = output_file_path
        self.number_of_bins = number_of_bins
        self.r_smooth = r_smooth
        self.alpha = alpha
        self.hsteps = hsteps
        self.bootstrap_flag = bootstrap_flag
        self.t_bootstrap = t_bootstrap
        self.r_bootstrap = r_bootstrap
        self.diagnostic_plots = diagnostic_plots
        self.eps_stop = eps_stop
        self.theta1 = theta1
        self.theta2 = theta2
        self.verbose = verbose
        self.noise_flag = noise_flag
        self.p_noise = p_noise
        self.savedata = savedata
        self.base_seed = base_seed
        
        # Initialize data-related attributes
        self.data = None
        self.ordered_data = None
        self.results = None
        
        # Store figure and axes as None initially
        self.fig = None
        self.axes = None
        
        # Fit data if provided
        if data is not None:
            self.fit(data)
            
        # Create the plots immediately if auto_plot is True and data is provided
        if auto_plot and data is not None:
            self.plot()
            
    def fit(self, data: np.ndarray) -> 'TailEstimatorSet':
        """Fit the estimators to the data.
        
        Parameters
        ----------
        data : np.ndarray
            The data to fit the estimators to.
            
        Returns
        -------
        self : TailEstimatorSet
            The fitted estimator set.
        """
        # Make sure data is a numpy array
        data_array = np.asarray(data)
        
        # Store the data
        self.data = data_array
        self.ordered_data = np.sort(data_array)[::-1]
        
        # Fit the estimators
        self.results = fit_estimators(
            ordered_data=self.ordered_data,
            number_of_bins=self.number_of_bins,
            r_smooth=self.r_smooth,
            alpha=self.alpha,
            hsteps=self.hsteps,
            bootstrap_flag=self.bootstrap_flag,
            t_bootstrap=self.t_bootstrap,
            r_bootstrap=self.r_bootstrap,
            diagn_plots=self.diagnostic_plots,
            eps_stop=self.eps_stop,
            verbose=self.verbose,
            noise_flag=self.noise_flag,
            p_noise=self.p_noise,
            base_seed=self.base_seed
        )
        
        # Reset figure and axes
        self.fig = None
        self.axes = None
        
        return self
    
    def plot(self) -> Tuple[plt.Figure, np.ndarray]:
        """Create and return the plots.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : numpy.ndarray
            Array of axes objects.
        """
        if self.ordered_data is None:
            raise ValueError("No data has been fitted. Call fit() first.")
            
        self.fig, self.axes = self._create_plots()
        return self.fig, self.axes
    
    def plot_diagnostics(self) -> Tuple[plt.Figure, np.ndarray]:
        """Create and return the diagnostic plots.
        
        Returns
        -------
        fig_d : matplotlib.figure.Figure
            The diagnostic figure object.
        axes_d : numpy.ndarray
            Array of diagnostic axes objects.
        
        Raises
        ------
        ValueError
            If no data has been fitted or if bootstrap is not enabled.
        """
        if self.ordered_data is None:
            raise ValueError("No data has been fitted. Call fit() first.")
            
        if not self.bootstrap_flag:
            raise ValueError("Diagnostic plots require bootstrap to be enabled. Set bootstrap_flag=True when creating the TailEstimatorSet.")
            
        if not self.diagnostic_plots:
            raise ValueError("Diagnostic plots are not enabled. Set diagnostic_plots=True when creating the TailEstimatorSet.")
            
        return self._create_diagnostic_plots()
    
    def _create_diagnostic_plots(self) -> Tuple[plt.Figure, np.ndarray]:
        """Create the diagnostic plots using the make_diagnostic_plots function.
        
        Returns
        -------
        fig_d : matplotlib.figure.Figure
            The diagnostic figure object.
        axes_d : numpy.ndarray
            Array of diagnostic axes objects.
        """
        from .plot.plot_methods import make_diagnostic_plots
        
        return make_diagnostic_plots(
            results=self.results,
            output_file_path=self.output_file_path,
            hsteps=self.hsteps,
            bootstrap_flag=self.bootstrap_flag,
            verbose=self.verbose,
            noise_flag=self.noise_flag,
            savedata=self.savedata,
        )
    
    def _create_plots(self) -> Tuple[plt.Figure, np.ndarray]:
        """Create the plots using the make_plots function.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : numpy.ndarray
            Array of axes objects.
        """
        return make_plots(
            ordered_data=self.ordered_data,
            results=self.results,
            output_file_path=self.output_file_path,
            alpha=self.alpha,
            bootstrap_flag=self.bootstrap_flag,
            diagn_plots=self.diagnostic_plots,
            theta1=self.theta1,
            theta2=self.theta2,
            verbose=self.verbose,
            noise_flag=self.noise_flag,
            savedata=self.savedata,
        )
    
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters used for plotting.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of parameters used for plotting.
        """
        return {
            'data_length': len(self.data) if self.data is not None else 0,
            'number_of_bins': self.number_of_bins,
            'r_smooth': self.r_smooth,
            'alpha': self.alpha,
            'hsteps': self.hsteps,
            'bootstrap_flag': self.bootstrap_flag,
            't_bootstrap': self.t_bootstrap,
            'r_bootstrap': self.r_bootstrap,
            'diagnostic_plots': self.diagnostic_plots,
            'eps_stop': self.eps_stop,
            'theta1': self.theta1,
            'theta2': self.theta2,
            'verbose': self.verbose,
            'noise_flag': self.noise_flag,
            'p_noise': self.p_noise,
            'savedata': self.savedata,
            'base_seed': self.base_seed
        }
    
    def __repr__(self) -> str:
        """Return a string representation of the object."""
        if self.data is None:
            return "TailEstimatorSet(not fitted)"
        else:
            return f"TailEstimatorSet(data_length={len(self.data)})"
    
    def __call__(self) -> Tuple[plt.Figure, np.ndarray]:
        """Return the figure and axes when the object is called."""
        if self.ordered_data is None:
            raise ValueError("No data has been fitted. Call fit() first.")
            
        if self.fig is None:
            self.plot()
        return self.fig, self.axes