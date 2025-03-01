"""Smooth Hill estimator implementation for tail index estimation."""
import numpy as np
from typing import Dict, Any, Tuple
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
        """Format Smooth Hill estimator parameters as a string.
        
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
        output += f"Smoothing window width (r): {self.r_smooth}\n"
        output += f"Number of order statistics: {len(params['k_arr'])}\n"
        output += f"Range of tail index estimates: [{min(params['xi_arr']):.4f}, {max(params['xi_arr']):.4f}]\n"
        
        return output