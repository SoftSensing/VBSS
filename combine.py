import numpy as np
import os
from tqdm import tqdm


# Function to safely load a batch file
def load_batch(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        return data['frames'], data['forces']
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


# Function to save combined data incrementally
def save_combined_data(frames_list, forces_list, file_path):
    if os.path.exists(file_path):
        existing_data = np.load(file_path, allow_pickle=True)
        combined_frames = np.concatenate((existing_data['frames'], np.concatenate(frames_list, axis=0)))
        combined_forces = np.concatenate((existing_data['forces'], np.concatenate(forces_list, axis=0)))
    else:
        combined_frames = np.concatenate(frames_list, axis=0)
        combined_forces = np.concatenate(forces_list, axis=0)

    np.savez(file_path, frames=combined_frames, forces=combined_forces)


# Total number of batches
num_batches = 38

# Collect all frames and forces for integrity check
all_frames = []
all_forces = []

# Main loop to process and concatenate data in chunks
chunk_size = 5  # Number of batches to process at once
combined_file_path = 'combined_data.npz'

for i in tqdm(range(0, num_batches, chunk_size), desc="Processing batches"):
    frames_list = []
    forces_list = []

    for j in range(i, min(i + chunk_size, num_batches)):
        batch_file = f'output_batch_{j}.npz'
        if os.path.exists(batch_file):
            frames, forces = load_batch(batch_file)
            if frames is not None and forces is not None:
                frames_list.append(frames)
                forces_list.append(forces)
                all_frames.append(frames)  # Save for integrity check
                all_forces.append(forces)  # Save for integrity check
                print(f"Batch {j}: frames shape = {frames.shape}, forces shape = {forces.shape}")
        else:
            print(f"File {batch_file} does not exist.")

    # Save combined data incrementally
    if frames_list and forces_list:
        save_combined_data(frames_list, forces_list, combined_file_path)

# Load and check the combined data
combined_data = np.load(combined_file_path)
combined_frames_c = combined_data['frames']
combined_forces_c = combined_data['forces']

print(f"Combined frames loaded shape: {combined_frames_c.shape}")
print(f"Combined forces loaded shape: {combined_forces_c.shape}")

# Verify data integrity after loading
frames_match = np.array_equal(np.concatenate(all_frames, axis=0), combined_frames_c)
forces_match = np.array_equal(np.concatenate(all_forces, axis=0), combined_forces_c)
assert frames_match, "Loaded frame data mismatch!"
assert forces_match, "Loaded force data mismatch!"

print("Data integrity check passed!")
