import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Initialize lists to store frames and forces from all batches
all_frames = []
all_forces = []

# Loop over the batch files
for i in range(0,11):  # Assuming you have batches from 0 to 11
    filen = f'output_batch_{i}.npz'
    data = np.load(filen)

    # Append frames and forces from this batch to the lists
    all_frames.append(data['frames'])
    all_forces.append(data['forces'])

# Concatenate all frames and forces arrays
all_frames = np.concatenate(all_frames, axis=0)
all_forces = np.concatenate(all_forces, axis=0)

# Plotting the frames
num_frames = all_frames.shape[0]
selected_indices = range(0, num_frames, 100)

rows = 3  # Adjust number of rows as needed
cols = 5  # Adjust number of columns as needed

plt.figure(figsize=(15, 10))
for i, idx in enumerate(selected_indices, start=1):
    if i > rows * cols:
        break  # Stop if we exceed the number of plots available
    frame_rgb = all_frames[idx][:, :, ::-1]  # Converting BGR to RGB
    plt.subplot(rows, cols, i)
    plt.imshow(frame_rgb)
    plt.title(f'Frame {idx}')
    plt.axis('off')
plt.tight_layout()
plt.show()


# Plotting forces data
force_labels = ['F_x', 'F_y', 'F_z', 'M_x', 'M_y', 'M_z', 'Temp']
plt.figure(figsize=(18, 12))
for i in range(7):
    plt.subplot(3, 3, i + 1)
    plt.plot(all_forces[:, i], label=f'{force_labels[i]}')
    plt.xlabel('Index')
    plt.ylabel(f'{force_labels[i]}')
    plt.title(f'{force_labels[i]} Distribution')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()

# Setting up the figure and axis for animation
fig, ax = plt.subplots()
img = ax.imshow(all_frames[0][:, :, ::-1])  # Display the first frame
ax.axis('off')  # Hide the axes

def update(frame_number):
    img.set_data(all_frames[frame_number][:, :, ::-1])  # Update the frame displayed
    return img,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(all_frames), blit=True, interval=50)  # Adjust interval for frame rate
ani.save('animation.gif', writer='ffmpeg', fps=20)