"""Pickands estimator implementation for tail index estimation."""
import numpy as np
from typing import Dict, Any, Tuple
from .result import TailEstimatorResult
from .base import BaseTailEstimator
from .tail_methods import pickands_estimator as pickands_estimate

class PickandsEstimator(BaseTailEstimator):
    """Pickands estimator for tail index estimation.
    
    This class implements the Pickands estimator, which is a simple method
    that does not use bootstrap procedures. Note that estimates can only be
    calculated up to the floor(n/4)-th order statistic.
    
    Parameters
    ----------
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
    
    def __init__(self, **kwargs):
        # Pickands estimator doesn't use bootstrap
        super().__init__(bootstrap=False, **kwargs)

    def _estimate(self, ordered_data: np.ndarray) -> Tuple:
        """Estimate the tail index using the Pickands estimator.
        
        Parameters
        ----------
        ordered_data : np.ndarray
            Data array in decreasing order.
            
        Returns
        -------
        Tuple
            Contains estimation results from pickands_estimator.
        """
        return pickands_estimate(ordered_data)

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