import deepSI
import numpy as np
import cv2
from deepSI import System_data
from matplotlib import pyplot as plt

# Load the npz file
filen = 'combined_data.npz'
data = np.load(filen, allow_pickle=True)  # Use np.load directly

# Assuming you have arrays 'frames' and 'forces' within your npz file
frames = data['frames']
forces = data['forces']

# Load the npz file
filen = 'combined_data.npz'
data = np.load(filen, allow_pickle=True)

# Determine the size of each set
total_size = len(frames)
train_size = int(total_size * 0.5)  # 50% of the data for training
val_size = int(total_size * 0.2)  # 20% of the data for validation
test_size = total_size - train_size - val_size  # Remaining 30% for testing

# Function to resize and reshape frames
def resize_and_reshape_frames(frames, batch_size, new_height, new_width):
    num_frames = frames.shape[0]
    resized_and_reshaped_frames = np.zeros((num_frames, frames.shape[3], new_height, new_width), dtype=np.uint8)
    for start in range(0, num_frames, batch_size):
        end = start + batch_size
        batch_frames = frames[start:end]
        for i in range(batch_frames.shape[0]):
            resized_frame = cv2.resize(batch_frames[i], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            resized_and_reshaped_frames[start + i] = resized_frame.transpose(2, 0, 1)
    return resized_and_reshaped_frames

# Resize and reshape parameters
new_height = frames.shape[1] // 4
new_width = frames.shape[2] // 4
batch_size = 30

# Resize and reshape all frames
frames_resized_reshaped = resize_and_reshape_frames(frames, batch_size, new_height, new_width)

# Split the data
frames_train = frames_resized_reshaped[:train_size]
frames_val = frames_resized_reshaped[train_size:train_size + val_size]
frames_test = frames_resized_reshaped[train_size + val_size:]

forces_train = forces[:train_size, 3]
forces_val = forces[train_size:train_size + val_size, 3]
forces_test = forces[train_size + val_size:, 3]

# Initialize the SS_encoder_CNN_video system
sys = deepSI.fit_systems.SS_encoder_CNN_video(na=20, nb=20)
n_channels, height, width = frames_train.shape[1], frames_train.shape[2], frames_train.shape[3]
sys.init_nets(nu=1, ny=(n_channels, height, width))

# Create System_data instances with resized and reshaped frames
system_data = System_data(u=forces_train, y=frames_train)
system_data_val = System_data(u=forces_val, y=frames_val)
system_data_test = System_data(u=forces_test, y=frames_test)

# Fit the model
sys.fit(system_data, val_sys_data=system_data_val, cuda=True, epochs=300, batch_size=64, loss_kwargs={'online_construct': True})

# Save the model
sys.save_system('model2')