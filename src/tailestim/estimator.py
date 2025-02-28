import numpy as np
from .tail_methods import (
    hill_estimator,
    moments_estimator,
    kernel_type_estimator,
    pickands_estimator,
    smooth_hill_estimator,
    add_uniform_noise
)

class TailEstimator:
    """A class for estimating tail parameters of heavy-tailed distributions.

    This class implements various methods for estimating the tail index (ξ) and related
    parameters of heavy-tailed distributions, particularly useful for power law analysis.

    Available Methods:
    - `hill`: Hill estimator with double-bootstrap for optimal threshold selection
    - `moments`: Moments estimator with double-bootstrap
    - `kernel`: Kernel-type estimator with double-bootstrap and bandwidth selection
    - `pickands`: Pickands estimator (no bootstrap)
    - `smooth_hill`: Smooth Hill estimator (no bootstrap)

    Parameters
    ----------
    method : str, default='hill'
        The estimation method to use. Must be one of:
        ['hill', 'moments', 'kernel', 'pickands', 'smooth_hill']
    bootstrap : bool, default=True
        Whether to use double-bootstrap for optimal threshold selection.
        Only applicable for 'hill', 'moments', and 'kernel' methods.
    **kwargs : dict
        Additional parameters passed to specific methods:
        - For 'kernel': hsteps (int, default=200), alpha (float, default=0.6)
        - For 'smooth_hill': r_smooth (int, default=2)

    Attributes
    ----------
    results : tuple or None
        Stores the estimation results after calling fit().
        The structure depends on the chosen method.

    Examples
    --------
    >>> import numpy as np
    >>> from tail_estimation import TailEstimator
    >>>
    >>> # Generate power law data
    >>> data = np.random.pareto(a=3, size=1000)
    >>>
    >>> # Fit using Hill estimator
    >>> estimator = TailEstimator(method='hill', bootstrap=True)
    >>> estimator.fit(data)
    >>>
    >>> # Get estimates
    >>> result = estimator.get_parameters()
    >>> print(f"Gamma: {result['gamma']:.3f}")
    """
    def __init__(self, method='hill', bootstrap=True, **kwargs):
        self.method = method
        self.bootstrap = bootstrap
        self.kwargs = kwargs
        self.results = None

    def fit(self, data):
        ordered_data = np.sort(data)[::-1]  # decreasing order required
        if self.method == 'hill':
            self.results = hill_estimator(ordered_data, bootstrap=self.bootstrap, **self.kwargs)
        elif self.method == 'moments':
            self.results = moments_estimator(ordered_data, bootstrap=self.bootstrap, **self.kwargs)
        elif self.method == 'kernel':
            hsteps = self.kwargs.get('hsteps', 200)
            alpha = self.kwargs.get('alpha', 0.6)
            self.results = kernel_type_estimator(ordered_data, hsteps, alpha=alpha, bootstrap=self.bootstrap, **self.kwargs)
        elif self.method == 'pickands':
            self.results = pickands_estimator(ordered_data)
        elif self.method == 'smooth_hill':
            r_smooth = self.kwargs.get('r_smooth', 2)
            self.results = smooth_hill_estimator(ordered_data, r_smooth=r_smooth)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def __str__(self):
        """Pretty print the estimation results."""
        if self.results is None:
            return "Model not fitted yet. Call fit() first."
        
        params = self.get_parameters()
        
        # Create header
        header = "=" * 50 + "\n"
        header += f"Tail Estimation Results ({self.method.title()} Method)\n"
        header += "=" * 50 + "\n\n"
        
        # Format main parameters
        main_params = "Parameters:\n"
        main_params += "-" * 20 + "\n"
        
        if self.method in ['hill', 'moments', 'kernel']:
            main_params += f"Optimal order statistic (k*): {params['k_star']:.0f}\n"
            main_params += f"Tail index (ξ): {params['xi_star']:.4f}\n"
            main_params += f"Gamma (powerlaw exponent) (γ): {params['gamma']:.4f}\n"
            
            if self.bootstrap:
                main_params += "\nBootstrap Results:\n"
                main_params += "-" * 20 + "\n"
                bs1 = params['bootstrap_results']['first_bootstrap']
                bs2 = params['bootstrap_results']['second_bootstrap']
                
                if self.method == 'kernel':
                    main_params += f"First bootstrap optimal bandwidth: {bs1['h_min']:.4f}\n"
                    main_params += f"Second bootstrap optimal bandwidth: {bs2['h_min']:.4f}\n"
                else:
                    main_params += f"First bootstrap minimum AMSE fraction: {bs1['k_min']:.4f}\n"
                    main_params += f"Second bootstrap minimum AMSE fraction: {bs2['k_min']:.4f}\n"
        
        elif self.method in ['pickands', 'smooth_hill']:
            main_params += "Note: This method provides arrays of estimates\n"
            main_params += f"Number of order statistics: {len(params['k_arr'])}\n"
            main_params += f"Range of tail index estimates: [{min(params['xi_arr']):.4f}, {max(params['xi_arr']):.4f}]\n"
        
        return header + main_params

    def get_parameters(self):
        """Get the estimated parameters."""
        if self.results is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Unpack results based on method
        if self.method in ['hill', 'moments']:
            k_arr, xi_arr, k_star, xi_star, x1_arr, n1_amse, k1, max_index1, x2_arr, n2_amse, k2, max_index2 = self.results
            gamma = 1 + 1./xi_star  # Calculate gamma
            return {
                'k_arr': k_arr,  # Array of order statistics
                'xi_arr': xi_arr,  # Array of tail index estimates
                'k_star': k_star,  # Optimal order statistic from double-bootstrap
                'xi_star': xi_star,  # Optimal tail index estimate
                'gamma': gamma,  # Powerlaw exponent gamma = 1 + 1/xi
                'bootstrap_results': {
                    'first_bootstrap': {
                        'x_arr': x1_arr,  # Fractions of order statistics
                        'amse': n1_amse,  # AMSE values
                        'k_min': k1,  # Minimum AMSE fraction
                        'max_index': max_index1  # Minimization boundary index
                    },
                    'second_bootstrap': {
                        'x_arr': x2_arr,
                        'amse': n2_amse,
                        'k_min': k2,
                        'max_index': max_index2
                    }
                } if self.bootstrap else None
            }
        elif self.method == 'kernel':
            k_arr, xi_arr, k_star, xi_star, x1_arr, n1_amse, h1, max_index1, x2_arr, n2_amse, h2, max_index2 = self.results
            gamma = 1 + 1./xi_star  # Calculate gamma
            return {
                'k_arr': k_arr,  # Array of order statistics
                'xi_arr': xi_arr,  # Array of tail index estimates
                'k_star': k_star,  # Optimal order statistic
                'xi_star': xi_star,  # Optimal tail index estimate
                'gamma': gamma,  # Extreme value index gamma = 1 + 1/xi
                'bootstrap_results': {
                    'first_bootstrap': {
                        'x_arr': x1_arr,  # Fractions of order statistics
                        'amse': n1_amse,  # AMSE values
                        'h_min': h1,  # Optimal bandwidth
                        'max_index': max_index1  # Minimization boundary index
                    },
                    'second_bootstrap': {
                        'x_arr': x2_arr,
                        'amse': n2_amse,
                        'h_min': h2,
                        'max_index': max_index2
                    }
                } if self.bootstrap else None
            }
        elif self.method in ['pickands', 'smooth_hill']:
            k_arr, xi_arr = self.results
            return {
                'k_arr': k_arr,  # Array of order statistics
                'xi_arr': xi_arr  # Array of tail index estimates
            }