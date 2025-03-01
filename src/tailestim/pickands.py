"""Pickands estimator implementation for tail index estimation."""
import numpy as np
from typing import Dict, Any, Tuple
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

    def get_parameters(self) -> Dict[str, Any]:
        """Get the estimated parameters.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - k_arr: Array of order statistics
            - xi_arr: Array of tail index estimates
        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        k_arr, xi_arr = self.results
        
        return {
            'k_arr': k_arr,
            'xi_arr': xi_arr
        }

    def _format_params(self, params: Dict[str, Any]) -> str:
        """Format Pickands estimator parameters as a string.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameters to format.
            
        Returns
        -------
        str
            Formatted parameter string.
        """
        output = "Note: This method provides arrays of estimates\n"
        output += f"Number of order statistics: {len(params['k_arr'])}\n"
        output += f"Range of tail index estimates: [{min(params['xi_arr']):.4f}, {max(params['xi_arr']):.4f}]\n"
        
        return output