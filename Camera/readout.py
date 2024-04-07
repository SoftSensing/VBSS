import matplotlib.pyplot as plt
import numpy as np

filen = 'output_batch_0.npz'
data = np.load(filen)
# Extract the frames from the npz file
frames = data['frames']

# Determine the number of frames and their shape
num_frames = frames.shape[0]
frame_shape = frames.shape[1:]

selected_indices = range(0, num_frames, 100)

# Create a figure for displaying the frames
plt.figure(figsize=(15, 10))

# Plot the selected frames
for i, idx in enumerate(selected_indices, start=1):
    frame_rgb = frames[idx][:, :, ::-1]

    plt.subplot(2, 5, i)
    plt.imshow(frame_rgb)
    plt.title(f'Frame {idx}')
    plt.axis('off')

# Show the plot
plt.tight_layout()
plt.show()

new_data = np.load(filen)

# Check if the new file has a weights array
if 'weights' in new_data.files:
    new_weights = new_data['weights']

    # Convert the Unicode string weights to numerical values (floats) for the new file
    new_weights_numerical = np.array([float(w) for w in new_weights])

    # Checking the first few entries of the new file
    new_weights_numerical_sample = new_weights_numerical[:10]
else:
    new_weights_numerical_sample = "No 'weights' array found in the new file."

plt.figure(figsize=(12, 6))
plt.plot(new_weights_numerical, label='Weight in kg', color='green')
plt.xlabel('Index')
plt.ylabel('Weight (kg)')
plt.title('New Weight Distribution')
plt.legend()
plt.grid(True)
plt.show()