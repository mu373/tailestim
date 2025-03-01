import os
import numpy as np

class TailData:
    def __init__(self, name):
        self.name = name
        self.data = self.load_data()

    def load_data(self):
        # Define the path to the data directory
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        file_path = os.path.join(data_dir, f'{self.name}.dat')

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file '{self.name}.dat' not found in package data directory.")

        # Load the data from the file using the provided method
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
        return f"TailData(name='{self.name}', data_length={len(self.data)})"