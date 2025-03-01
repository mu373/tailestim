"""Base class for tail index estimation."""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class BaseTailEstimator(ABC):
    """Abstract base class for tail index estimation.

    This class defines the common interface and utility methods for all tail
    estimation implementations. Each specific estimation method should inherit
    from this class and implement the required abstract methods.

    Parameters
    ----------
    bootstrap : bool, default=True
        Whether to use double-bootstrap for optimal threshold selection.
        May not be applicable for all methods.
    **kwargs : dict
        Additional parameters specific to each estimation method.
    """
    
    def __init__(self, bootstrap: bool = True, **kwargs):
        self.bootstrap = bootstrap
        self.kwargs = kwargs
        self.results = None

    @abstractmethod
    def _estimate(self, ordered_data: np.ndarray) -> Tuple:
        """Core estimation method to be implemented by each specific estimator.
        
        Parameters
        ----------
        ordered_data : np.ndarray
            Data array in decreasing order.
            
        Returns
        -------
        Tuple
            Estimation results specific to each method.
        """
        pass

    def fit(self, data: np.ndarray) -> None:
        """Fit the estimator to the data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data array.
        """
        ordered_data = np.sort(data)[::-1]  # decreasing order required
        self.results = self._estimate(ordered_data)

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get the estimated parameters.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing the estimated parameters.
            The structure depends on the specific estimation method.
        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return {}

    def __str__(self) -> str:
        """Format estimation results as a string."""
        if self.results is None:
            return "Model not fitted yet. Call fit() first."
        
        params = self.get_parameters()
        
        # Create header
        header = "=" * 50 + "\n"
        header += f"Tail Estimation Results ({self.__class__.__name__})\n"
        header += "=" * 50 + "\n\n"
        
        # Format main parameters
        main_params = "Parameters:\n"
        main_params += "-" * 20 + "\n"
        
        # Add method-specific parameter formatting
        params_str = self._format_params(params)
        
        return header + main_params + params_str

    @abstractmethod
    def _format_params(self, params: Dict[str, Any]) -> str:
        """Format method-specific parameters as a string.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameters to format.
            
        Returns
        -------
        str
            Formatted parameter string.
        """
        pass