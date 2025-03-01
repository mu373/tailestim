import os
import pytest
import numpy as np
from tailestim.datasets import TailData

def test_load_existing_data():
    """Test loading an existing dataset"""
    # Test with CAIDA dataset
    data = TailData("CAIDA_KONECT")
    assert isinstance(data.data, np.ndarray)
    assert len(data.data) > 0
    
    # Test with Libimseti dataset
    data = TailData("Libimseti_in_KONECT")
    assert isinstance(data.data, np.ndarray)
    assert len(data.data) > 0
    
    # Test with Pareto dataset
    data = TailData("Pareto")
    assert isinstance(data.data, np.ndarray)
    assert len(data.data) > 0

def test_nonexistent_data():
    """Test handling of non-existent dataset"""
    with pytest.raises(FileNotFoundError):
        TailData("nonexistent_dataset")

def test_data_format():
    """Test if data is properly formatted"""
    data = TailData("CAIDA_KONECT")
    
    # Data should be a numpy array
    assert isinstance(data.data, np.ndarray)
    
    # All values should be numeric
    assert np.issubdtype(data.data.dtype, np.number)
    
def test_representation():
    """Test string representation of TailData"""
    data = TailData("CAIDA_KONECT")
    repr_str = repr(data)
    
    # Check if representation contains the name
    assert "CAIDA_KONECT" in repr_str
    
    # Check if representation contains the data length
    assert str(len(data.data)) in repr_str
    
    # Check format
    assert repr_str.startswith("TailData(")
    assert repr_str.endswith(")")

def test_data_consistency():
    """Test if loaded data is consistent with file content"""
    data = TailData("CAIDA_KONECT")
    
    # Get the data file path
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'tailestim', 'data')
    file_path = os.path.join(examples_dir, 'CAIDA_KONECT.dat')
    
    # Read the file manually to verify data
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Calculate total count from file
    total_count = sum(int(line.strip().split()[1]) for line in lines)
    
    # Verify the loaded data length matches the file content
    assert len(data.data) == total_count
    
    # Verify some values from the file match the loaded data
    first_line = lines[0].strip().split()
    first_value = float(first_line[0])
    first_count = int(first_line[1])
    assert np.all(data.data[:first_count] == first_value)