import os
import numpy as np
import logging

class TailData:
    """Load and manage tail distribution datasets.
    
    This class provides functionality to load datasets either from the package's
    built-in data directory using a name, or from a custom path provided by the user.
    
    Parameters
    ----------
    name : str, optional
        Name of a built-in dataset to load (without file extension).
        Must be provided if `path` is None.
    path : str, optional
        Path to a custom dataset file. If provided, this takes precedence over `name`.
        Must be provided if `name` is None.
        
    Attributes
    ----------
    name : str or None
        Name of the dataset if a built-in dataset was loaded.
    path : str or None
        Path to the dataset file if a custom dataset was loaded.
    data : numpy.ndarray
        The loaded dataset as a numpy array.
        
    Examples
    --------
    Load a built-in dataset:
    
    >>> data = TailData(name='CAIDA_KONECT')
    >>> print(len(data.data))
    
    Load a custom dataset:
    
    >>> data = TailData(path='path/to/my/data.dat')
    >>> print(len(data.data))
    """
    
    def __init__(self, name=None, path=None):
        if name is None and path is None:
            raise ValueError("Either 'name' or 'path' must be provided")
        
        if name is not None and path is not None:
            logging.info("Both 'name' and 'path' provided; 'path' will take precedence")
        
        self.name = name
        self.path = path
        self.data = self.load_data()

    def load_data(self):
        """Load data from either a built-in dataset or a custom file path.
        
        Returns
        -------
        numpy.ndarray
            The loaded dataset as a numpy array.
            
        Raises
        ------
        FileNotFoundError
            If the specified dataset file cannot be found.
        """
        # Determine the file path based on whether name or path was provided
        if self.path is not None:
            # Use the provided custom path
            file_path = self.path
            logging.info(f"Using custom path: {file_path}")
        else:
            # Use the package data directory with the provided name
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
            file_path = os.path.join(data_dir, f'{self.name}.dat')
            logging.info(f"Using package data path: {file_path}")

        # Check if the file exists
        if not os.path.exists(file_path):
            if self.path is not None:
                raise FileNotFoundError(f"Data file not found at path: {file_path}")
            else:
                raise FileNotFoundError(f"Data file '{self.name}.dat' not found in package data directory.")

        # Load the data from the file using the provided method
        logging.info(f"Loading data from file: {file_path}")
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Determine the total number of data points
        N = sum(int(line.strip().split()[1]) for line in lines)
        ordered_data = np.zeros(N)
        current_index = 0

        # Populate the ordered_data array
        for line in lines:
            degree, count = line.strip().split()
            ordered_data[current_index:current_index + int(count)] = float(degree)
            current_index += int(count)

        return ordered_data

    def __repr__(self):
        """Return a string representation of the TailData object.
        
        Returns
        -------
        str
            String representation including the data source and length.
        """
        if self.path is not None:
            return f"TailData(path='{self.path}', data_length={len(self.data)})"
        else:
            return f"TailData(name='{self.name}', data_length={len(self.data)})"