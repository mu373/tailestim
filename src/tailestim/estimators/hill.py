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

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the estimator.
        
        Returns
        -------
        dict
            Dictionary containing the parameters of the estimator.
        """
        return {
            "base_seed": self.base_seed,
            "bootstrap": self.bootstrap,
            "t_bootstrap": self.t_bootstrap,
            "r_bootstrap": self.r_bootstrap,
            "eps_stop": self.eps_stop,
            "verbose": self.verbose,
            "diagn_plots": self.diagn_plots,
            **self.kwargs
        }

    def get_result(self) -> TailEstimatorResult:
        """Get the estimated parameters.

        Attributes
        ----------
        estimator : BaseTailEstimator
            The estimator instance (e.g., HillEstimator, PickandsEstimator, etc.) used for estimation.
        xi_star_ : float
            Optimal tail index estimate (ξ).
        gamma_ : float
            Power law exponent (γ).
        k_arr_ : np.ndarray
            Array of order statistics.
        xi_arr_ : np.ndarray
            Array of tail index estimates.
        k_star_ : float
            Optimal order statistic (k*).
        bootstrap_results_ : dict
            Bootstrap results.

        Returns
        -------
        TailEstimatorResult
        """

        if self.results is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        k_arr, xi_arr, k_star, xi_star, x1_arr, n1_amse, k1, max_index1, \
        x2_arr, n2_amse, k2, max_index2 = self.results
        
        res = {
            'k_arr_': k_arr,
            'xi_arr_': xi_arr,
        }
        
        if self.bootstrap and k_star is not None:
            gamma = 1 + 1./xi_star
            res.update({
                'estimator': self,
                'k_star_': k_star,
                'xi_star_': xi_star,
                'gamma_': gamma,
                'bootstrap_results_': {
                    'first_bootstrap_': {
                        'x_arr_': x1_arr,
                        'amse_': n1_amse,
                        'k_min_': k1,
                        'max_index_': max_index1
                    },
                    'second_bootstrap_': {
                        'x_arr_': x2_arr,
                        'amse_': n2_amse,
                        'k_min_': k2,
                        'max_index_': max_index2
                    }
                }
            })
        
        return TailEstimatorResult(res)
