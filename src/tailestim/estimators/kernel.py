"""Kernel-type estimator implementation for tail index estimation."""
import numpy as np
from typing import Dict, Any, Tuple, Union
from numpy.random import BitGenerator, SeedSequence, RandomState, Generator
from .base import BaseTailEstimator
from .result import TailEstimatorResult
from .tail_methods import kernel_type_estimator as kernel_estimate

class KernelTypeEstimator(BaseTailEstimator):
    """Kernel-type estimator for tail index estimation.
    
    This class implements the Kernel-type estimator with optional double-bootstrap
    for optimal bandwidth selection. It uses both biweight and triweight kernels
    for estimation.
    
    Parameters
    ----------
    bootstrap : bool, default=True
        Whether to use double-bootstrap for optimal threshold selection.
    hsteps : int, default=200
        Parameter controlling number of bandwidth steps.
    alpha : float, default=0.6
        Parameter controlling the amount of "smoothing".
        Should be greater than 0.5.
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
        hsteps: int = 200,
        alpha: float = 0.6,
        t_bootstrap: float = 0.5,
        r_bootstrap: int = 500,
        eps_stop: float = 0.99,
        verbose: bool = False,
        diagn_plots: bool = False,
        base_seed: Union[None, SeedSequence, BitGenerator, Generator, RandomState] = None,
        **kwargs
    ):
        super().__init__(bootstrap=bootstrap, base_seed=base_seed, **kwargs)
        self.hsteps = hsteps
        self.alpha = alpha
        self.t_bootstrap = t_bootstrap
        self.r_bootstrap = r_bootstrap
        self.eps_stop = eps_stop
        self.verbose = verbose
        self.diagn_plots = diagn_plots

    def _estimate(self, ordered_data: np.ndarray) -> Tuple:
        """Estimate tail index using kernel-type estimator.
        
        Parameters
        ----------
        ordered_data : np.ndarray
            Data array in decreasing order.
            
        Returns
        -------
        Tuple
            Contains estimation results from kernel_type_estimator.
        """
        return kernel_estimate(
            ordered_data,
            self.hsteps,
            alpha=self.alpha,
            bootstrap=self.bootstrap,
            t_bootstrap=self.t_bootstrap,
            r_bootstrap=self.r_bootstrap,
            verbose=self.verbose,
            diagn_plots=self.diagn_plots,
            eps_stop=self.eps_stop,
            base_seed=self.base_seed
        )

    def get_result(self) -> TailEstimatorResult:
        """Get the estimated parameters.

        Attributes
        ----------
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
        
        k_arr, xi_arr, k_star, xi_star, x1_arr, n1_amse, h1, max_index1, \
        x2_arr, n2_amse, h2, max_index2 = self.results
        
        params = {
            'k_arr_': k_arr,
            'xi_arr_': xi_arr,
        }
        
        if self.bootstrap and k_star is not None:
            gamma = float('inf') if xi_star <= 0 else 1 + 1./xi_star
            params.update({
                'k_star_': k_star,
                'xi_star_': xi_star,
                'gamma_': gamma,
                'bootstrap_results_': {
                    'first_bootstrap_': {
                        'x_arr_': x1_arr,
                        'amse_': n1_amse,
                        'h_min_': h1,
                        'max_index_': max_index1
                    },
                    'second_bootstrap_': {
                        'x_arr_': x2_arr,
                        'amse_': n2_amse,
                        'h_min_': h2,
                        'max_index_': max_index2
                    }
                }
            })
        
        return TailEstimatorResult(params)

    def _format_params(self, params: Dict[str, Any]) -> str:
        """Format Kernel-type estimator parameters as a string.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameters to format.
            
        Returns
        -------
        str
            Formatted parameter string.
        """
        output = ""
        
        if hasattr(params, 'k_star_'):
            output += f"Optimal order statistic (k*): {params.k_star_:.0f}\n"
            output += f"Tail index (ξ): {params.xi_star_:.4f}\n"
            if params.gamma_ == float('inf'):
                output += "Gamma (powerlaw exponent) (γ): infinity (ξ <= 0)\n"
            else:
                output += f"Gamma (powerlaw exponent) (γ): {params.gamma_:.4f}\n"
            
            if self.bootstrap:
                output += "\nBootstrap Results:\n"
                output += "-" * 20 + "\n"
                bs1 = params.bootstrap_results_.first_bootstrap_
                bs2 = params.bootstrap_results_.second_bootstrap
                output += f"First bootstrap optimal bandwidth: {bs1.h_min:.4f}\n"
                output += f"Second bootstrap optimal bandwidth: {bs2.h_min:.4f}\n"
        else:
            output += "Note: No bootstrap results available\n"
            output += f"Number of order statistics: {len(params.k_arr_)}\n"
            output += f"Range of tail index estimates: [{min(params.xi_arr_):.4f}, {max(params.xi_arr_):.4f}]\n"
        
        return output