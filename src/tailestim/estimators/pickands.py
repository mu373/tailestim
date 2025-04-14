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

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the estimator.
        
        Returns
        -------
        dict
            Dictionary containing the parameters of the estimator.
        """
        return {
            **self.kwargs
        }

    def get_result(self) -> TailEstimatorResult:
        """Get the estimated parameters.
        
        Attributes
        ----------
        estimator : BaseTailEstimator
            The estimator instance (e.g., HillEstimator, PickandsEstimator, etc.) used for estimation.
        k_arr_ : np.ndarray
            Array of order statistics.
        xi_arr_ : np.ndarray
            Array of tail index estimates.

        Returns
        -------
        TailEstimatorResult
        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        k_arr, xi_arr = self.results
        
        res = {
            'estimator': self,
            'k_arr_': k_arr,
            'xi_arr_': xi_arr
        }
        
        return TailEstimatorResult(res)