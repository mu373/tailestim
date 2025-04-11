"""Smooth Hill estimator implementation for tail index estimation."""
import numpy as np
from typing import Dict, Any, Tuple
from .result import TailEstimatorResult
from .base import BaseTailEstimator
from .tail_methods import smooth_hill_estimator as smooth_hill_estimate

class SmoothHillEstimator(BaseTailEstimator):
    """Smooth Hill estimator for tail index estimation.
    
    This class implements the Smooth Hill estimator, which applies smoothing
    to the classical Hill estimator. It does not use bootstrap procedures.
    
    Parameters
    ----------
    r_smooth : int, default=2
        Integer parameter controlling the width of smoothing window.
        Typically small value such as 2 or 3.
    **kwargs : dict
        Additional parameters (not used by this estimator).

    Attributes
    ----------
    results : tuple or None
        Stores the estimation results after calling fit().
        Contains:
        - k_arr: Array of order statistics
        - xi_arr: Array of tail index estimates
    """
    
    def __init__(self, r_smooth: int = 2, **kwargs):
        # Smooth Hill estimator doesn't use bootstrap
        super().__init__(bootstrap=False, **kwargs)
        self.r_smooth = r_smooth

    def _estimate(self, ordered_data: np.ndarray) -> Tuple:
        """Estimate the tail index using the Smooth Hill method.
        
        Parameters
        ----------
        ordered_data : np.ndarray
            Data array in decreasing order.
            
        Returns
        -------
        Tuple
            Contains estimation results from smooth_hill_estimator.
        """
        return smooth_hill_estimate(ordered_data, r_smooth=self.r_smooth)

    def get_parameters(self) -> TailEstimatorResult:
        """Get the estimated parameters.
                
        Attributes
        ----------
        k_arr : np.ndarray
            Array of order statistics.
        xi_arr : np.ndarray
            Array of tail index estimates.

        Returns
        -------
        TailEstimatorResult
        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        k_arr, xi_arr = self.results
        
        params = {
            'k_arr': k_arr,
            'xi_arr': xi_arr
        }
        
        return TailEstimatorResult(params)