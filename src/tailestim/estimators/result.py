from typing import Dict, Any
import numpy as np


class TailEstimatorResult:
    """
    Class for storing the results of a tail estimator. Attributes available depends on the estimator used.

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
    k_min_ : float
        Minimum AMSE fraction.
    amse_ : np.ndarray
        AMSE values.
    max_index_ : int
        Maximum index.
    x_arr_ : np.ndarray
        Fraction of order statistics.
    """

    # Mapping of keys to human-readable labels
    _key_labels = {
        'estimator': 'Estimator',
        'xi_star_': 'Tail index (ξ)',
        'gamma_': 'Power law exponent (γ)',
        'k_arr_': 'Order statistics',
        'xi_arr_': 'Tail index estimates',
        'k_star_': 'Optimal order statistic (k*)',
        'bootstrap_results_': 'Bootstrap Results',
        'first_bootstrap_': 'First Bootstrap',
        'second_bootstrap_': 'Second Bootstrap',
        'k_min_': 'Minimum AMSE fraction',
        'amse_': 'AMSE values',
        'max_index_': 'Maximum index',
        'x_arr_': 'Fraction of order statistics'
    }
    
    def __init__(self, initial_data: Dict[str, Any] = None, **kwargs):
        data = initial_data or {}
        data.update(kwargs)
        # Recursively convert nested dicts to TailEstimatorResult objects.
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = TailEstimatorResult(value)
        self.__dict__.update(data)
    
    def __str__(self, include_header=True) -> str:
        """
        Return a human-readable string representation of the TailEstimatorResult.
        
        Parameters
        ----------
        include_header : bool, default=True
            Whether to include the "Result" header. This is set to False for nested objects.
        """
        output = []
        from .base import BaseTailEstimator
        
        # Add a header for the result section, but only for the top-level object
        if include_header:
            output.extend(["-"* 50, "Result", "-"* 50])
        
        # Order the attributes based on the keys in _key_labels
        ordered_keys = [k for k in self._key_labels if k in self.__dict__]
        # Add any keys that aren't in _key_labels at the end
        ordered_keys.extend([k for k in self.__dict__ if k not in self._key_labels])

        for key in ordered_keys:
            value = self.__dict__[key]
            label = self._key_labels.get(key, key.replace('_', ' ').title())

            # Format the value based on its type
            if isinstance(value, BaseTailEstimator):
                # Print estimator type
                value_str = f"{value.__class__.__name__}"
            elif isinstance(value, np.ndarray):
                if len(value) > 5:
                    # For large arrays, show shape and a few values
                    value_str = f"Array of shape {value.shape} [{', '.join(f'{v:.4f}' for v in value[:3])}, ...]"
                else:
                    # For small arrays, show all values
                    value_str = f"[{', '.join(f'{v:.4f}' for v in value)}]"
            elif isinstance(value, float):
                value_str = f"{value:.4f}"
            elif isinstance(value, TailEstimatorResult):
                # For nested TailEstimatorResult objects, indent their string representation
                # but don't include the header
                nested_str = value.__str__(include_header=False).replace('\n', '\n  ')
                value_str = f"\n  {nested_str}"
            else:
                value_str = str(value)
            
            output.append(f"{label}: {value_str}")
        
        return '\n'.join(output)

    def __repr__(self) -> str:
        """
        Return a string representation of the TailEstimatorResult.
        """
        return f"TailEstimatorResult(estimator={self.estimator.__class__.__name__})"