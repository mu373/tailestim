"""Hill estimator implementation for tail index estimation."""
import numpy as np
from typing import Dict, Any, Tuple, Union
from numpy.random import BitGenerator, SeedSequence, RandomState, Generator
from .base import BaseTailEstimator
from .result import TailEstimatorResult
from .tail_methods import hill_estimator as hill_estimate

class HillEstimator(BaseTailEstimator):
    """Hill estimator for tail index estimation.
    
    This class implements the Hill estimator with optional double-bootstrap
    for optimal threshold selection.
    
    Parameters
    ----------
    bootstrap : bool, default=True
        Whether to use double-bootstrap for optimal threshold selection.
    t_bootstrap : float, default=0.5
        Parameter controlling the size of the 2nd bootstrap.
        Defined from n2 = n*(t_bootstrap).
    r_bootstrap : int, default=500
        Number of bootstrap resamplings for the 1st and 2nd bootstraps.
    eps_stop : float, default=0.99
        Parameter controlling range of AMSE minimization.
        Defined as the fraction of order statistics to consider
        during the AMSE minimization step.
    verbose : bool, default=False
        Flag controlling bootstrap verbosity.
    diagn_plots : bool, default=False
        Flag to switch on/off generation of AMSE diagnostic plots.
    base_seed: None | SeedSequence | BitGenerator | Generator | RandomState, default=None
        Base random seed for reproducibility of bootstrap.
    """
    
    def __init__(
        self,
        bootstrap: bool = True,
        t_bootstrap: float = 0.5,
        r_bootstrap: int = 500,
        eps_stop: float = 0.99,
        verbose: bool = False,
        diagn_plots: bool = False,
        base_seed: Union[None, SeedSequence, BitGenerator, Generator, RandomState] = None,
        **kwargs
    ):
        super().__init__(bootstrap=bootstrap, base_seed=base_seed, **kwargs)
        self.t_bootstrap = t_bootstrap
        self.r_bootstrap = r_bootstrap
        self.eps_stop = eps_stop
        self.verbose = verbose
        self.diagn_plots = diagn_plots

    def _estimate(self, ordered_data: np.ndarray) -> Tuple:
        """Estimate the tail index using the Hill estimator.
        
        Parameters
        ----------
        ordered_data : np.ndarray
            Data array in decreasing order.
            
        Returns
        -------
        Tuple
            Contains estimation results from hill_estimator.
        """
        return hill_estimate(
            ordered_data,
            bootstrap=self.bootstrap,
            t_bootstrap=self.t_bootstrap,
            r_bootstrap=self.r_bootstrap,
            verbose=self.verbose,
            diagn_plots=self.diagn_plots,
            eps_stop=self.eps_stop,
            base_seed=self.base_seed
        )

    def get_parameters(self) -> TailEstimatorResult:
        """Get the estimated parameters.

        Attributes
        ----------
        xi_star : float
            Optimal tail index estimate (ξ).
        gamma : float
            Power law exponent (γ).
        k_arr : np.ndarray
            Array of order statistics.
        xi_arr : np.ndarray
            Array of tail index estimates.
        k_star : float
            Optimal order statistic (k*).
        bootstrap_results : dict
            Bootstrap results.

        Returns
        -------
        TailEstimatorResult
        """

        if self.results is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        k_arr, xi_arr, k_star, xi_star, x1_arr, n1_amse, k1, max_index1, \
        x2_arr, n2_amse, k2, max_index2 = self.results
        
        params = {
            'k_arr': k_arr,
            'xi_arr': xi_arr,
        }
        
        if self.bootstrap and k_star is not None:
            gamma = 1 + 1./xi_star
            params.update({
                'k_star': k_star,
                'xi_star': xi_star,
                'gamma': gamma,
                'bootstrap_results': {
                    'first_bootstrap': {
                        'x_arr': x1_arr,
                        'amse': n1_amse,
                        'k_min': k1,
                        'max_index': max_index1
                    },
                    'second_bootstrap': {
                        'x_arr': x2_arr,
                        'amse': n2_amse,
                        'k_min': k2,
                        'max_index': max_index2
                    }
                }
            })
        
        return TailEstimatorResult(params)
