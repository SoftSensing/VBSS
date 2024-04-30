import numpy as np

# List of file names to combine
file_names = [f'output_batch_{i}.npz' for i in range(0,11)]  # Adjust the range according to your files

# Dictionary to hold the data from all files
combined_data = {}

# Load each file and combine the data
for file_name in file_names:
    with np.load(file_name) as data:
        for key in data.files:
            if key in combined_data:
                # Append to the existing array if the key already exists
                combined_data[key] = np.concatenate((combined_data[key], data[key]))
            else:
                # Create a new entry in the dictionary if the key does not exist
                combined_data[key] = data[key]

# Save the combined data into a single .npz file
np.savez('combined_data.npz', **combined_data)
