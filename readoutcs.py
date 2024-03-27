import matplotlib.pyplot as plt
import numpy as np

filen = 'output_batch_1.npz'
data = np.load(filen)
frames = data['frames']

# Plotting the frames
num_frames = frames.shape[0]
frame_shape = frames.shape[1:]
selected_indices = range(0, num_frames, 100)
plt.figure(figsize=(15, 10))
for i, idx in enumerate(selected_indices, start=1):
    frame_rgb = frames[idx][:, :, ::-1]
    plt.subplot(2, 5, i)
    plt.imshow(frame_rgb)
    plt.title(f'Frame {idx}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Plotting forces data
forces_data = data['forces']

# Create a subplot for each force component
force_labels = ['F_x', 'F_y', 'F_z', 'M_x', 'M_y', 'M_z', 'Temp']
plt.figure(figsize=(18, 12))
for i in range(7):
    plt.subplot(3, 3, i+1)
    plt.plot(forces_data[:, i], label=f'{force_labels[i]}')
    plt.xlabel('Index')
    plt.ylabel(f'{force_labels[i]}')
    plt.title(f'{force_labels[i]} Distribution')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()
