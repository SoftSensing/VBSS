import numpy as np

# Define the number of batch files
num_files = 10  # Adjust this number to the total number of files you have

# Prepare a list to hold the data from each key found in the first file
data_dict = {}

# Initialize data structures by inspecting the first file
first_file_path = 'output_batch_0.npz'  # Adjust if files are located differently
with np.load(first_file_path, allow_pickle=True) as data:
    for key in data.files:
        data_dict[key] = [data[key]]  # Start a list for each key

# Process the remaining files
for i in range(1, num_files):
    file_path = f'output_batch_{i}.npz'  # Adjust the path if needed
    with np.load(file_path, allow_pickle=True) as data:
        for key in data.files:
            data_dict[key].append(data[key])  # Append data for each key

# Combine all arrays for each key into single arrays
for key in data_dict:
    data_dict[key] = np.concatenate(data_dict[key], axis=0)

# Optionally, save the combined arrays to a new file
np.savez_compressed('combined_output_20_3_new.npz', **data_dict)

print("Data combined successfully. Data available under the following keys:", list(data_dict.keys()))
