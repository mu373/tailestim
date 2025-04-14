"""Base class for tail index estimation."""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union
from numpy.random import BitGenerator, SeedSequence, RandomState, Generator
from .result import TailEstimatorResult

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
    base_seed: None | SeedSequence | BitGenerator | Generator | RandomState, default=None
        Base random seed for reproducibility of bootstrap. Only used for methods with bootstrap.
    **kwargs : dict
        Additional parameters specific to each estimation method.
    """
    
    def __init__(
            self,
            bootstrap: bool = True,
            base_seed: Union[None, SeedSequence, BitGenerator, Generator, RandomState] = None,
            **kwargs):
        self.bootstrap = bootstrap
        self.base_seed = base_seed
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
            Input data array (e.g., degree sequence). The data will automatically be sorted in decreasing order.
        """
        ordered_data = np.sort(data)[::-1]  # Each estimating functions require the data to be in decreasing order
        self.results = self._estimate(ordered_data)

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the estimator.
        
        Returns
        -------
        dict
            Dictionary containing the parameters of the estimator.
        """
        return {
            "bootstrap": self.bootstrap,
            "base_seed": self.base_seed,
            **self.kwargs
        }

    @abstractmethod
    def get_result(self) -> TailEstimatorResult:
        """Get the estimated parameters.
        
        Returns
        -------
        TailEstimatorResult
            Object containing the estimated parameters. Parameters and results included varies by method.

        Examples
        --------
        >>> hill = HillEstimator()
        >>> hill.fit(data)
        >>> result = hill.get_result()
        >>> gamma = result.gamma_
        >>> xi = result.xi_

        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return TailEstimatorResult()
    
    def __repr__(self) -> str:
        """Return a string representation of the estimator."""
        return f"{self.__class__.__name__}(bootstrap={self.bootstrap}, base_seed={self.base_seed}, kwargs={self.kwargs})"

    def __str__(self) -> str:
        """Format estimation object as a string."""
        # Create a string with the estimator type and fitted status
        estim_str = "-" * 50 + "\n"
        estim_str += f"Estimator Type: {self.__class__.__name__}\n"
        estim_str += "-" * 50 + "\n"
        estim_str += f"Fitted: {'Yes' if self.results is not None else 'No'}\n"
        
        # Add the arguments provided
        estim_str += "Arguments:\n"
        estim_str += f"  bootstrap: {self.bootstrap}\n"
        estim_str += f"  base_seed: {self.base_seed}\n"
        
        # Add any additional kwargs
        if self.kwargs:
            for key, value in self.kwargs.items():
                estim_str += f"  {key}: {value}\n"
        
        # If the model is not fitted, return just the estim_str 
        if self.results is None:
            return estim_str + "Model not fitted yet. Call fit() first."
        
        return estim_str 
