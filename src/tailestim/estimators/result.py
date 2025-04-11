from typing import Dict, Any
import numpy as np

class TailEstimatorResult:
    """
    Class for storing the results of a tail estimator. Attributes available depends on the estimator used.

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
    k_min : float
        Minimum AMSE fraction.
    amse : np.ndarray
        AMSE values.
    max_index : int
        Maximum index.
    x_arr : np.ndarray
        Fraction of order statistics.
    """

    # Mapping of keys to human-readable labels
    _key_labels = {
        'k_arr': 'Order statistics',
        'xi_arr': 'Tail index estimates',
        'k_star': 'Optimal order statistic (k*)',
        'xi_star': 'Tail index (ξ)',
        'gamma': 'Power law exponent (γ)',
        'bootstrap_results': 'Bootstrap Results',
        'first_bootstrap': 'First Bootstrap',
        'second_bootstrap': 'Second Bootstrap',
        'k_min': 'Minimum AMSE fraction',
        'amse': 'AMSE values',
        'max_index': 'Maximum index',
        'x_arr': 'Fraction of order statistics'
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
        
        # Add a header for the result section, but only for the top-level object
        if include_header:
            output.extend(["-"* 50, "Result", "-"* 50])
        
        # Format each attribute
        for key, value in self.__dict__.items():
            label = self._key_labels.get(key, key.replace('_', ' ').title())
            
            # Format the value based on its type
            if isinstance(value, np.ndarray):
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
        return f"TailEstimatorResult({self.__dict__})"