import os
import pytest
import numpy as np
import tempfile
from tailestim.datasets import TailData

def test_load_existing_data():
    """Test loading an existing dataset"""
    # Test with CAIDA dataset
    data = TailData(name="CAIDA_KONECT")
    assert isinstance(data.data, np.ndarray)
    assert len(data.data) > 0
    
    # Test with Libimseti dataset
    data = TailData(name="Libimseti_in_KONECT")
    assert isinstance(data.data, np.ndarray)
    assert len(data.data) > 0
    
    # Test with Pareto dataset
    data = TailData(name="Pareto")
    assert isinstance(data.data, np.ndarray)
    assert len(data.data) > 0

def test_nonexistent_data():
    """Test handling of non-existent dataset"""
    with pytest.raises(FileNotFoundError):
        TailData(name="nonexistent_dataset")
        
def test_missing_parameters():
    """Test handling of missing parameters"""
    with pytest.raises(ValueError):
        TailData()

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
    data = TailData(name="CAIDA_KONECT")
    
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
    
def test_data_load_custom_path():
    """Test loading data from a custom path"""
    # Create a temporary file with test data
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("10 3\n")
        temp_file.write("20 2\n")
        temp_file.write("30 1\n")
        temp_path = temp_file.name
    
    try:
        # Load data from the custom path
        data = TailData(path=temp_path)
        
        # Verify the data was loaded correctly
        assert isinstance(data.data, np.ndarray)
        assert len(data.data) == 6  # 3 + 2 + 1 = 6
        assert np.all(data.data[:3] == 10)
        assert np.all(data.data[3:5] == 20)
        assert data.data[5] == 30
        
        # Check the string representation
        repr_str = repr(data)
        assert temp_path in repr_str
        assert "data_length=6" in repr_str
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)
        
def test_path_precedence():
    """Test that path takes precedence over name when both are provided"""
    # Create a temporary file with test data
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("100 1\n")
        temp_path = temp_file.name
    
    try:
        # Load data with both name and path
        data = TailData(name="CAIDA_KONECT", path=temp_path)
        
        # Verify that the path was used, not the name
        assert len(data.data) == 1
        assert data.data[0] == 100
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)