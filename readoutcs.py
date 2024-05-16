import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Load the batch files (simplified to one file for demonstration)
filen = 'combined_data_dome1.npz'  # Update as necessary
data = np.load(filen)
all_frames = data['frames']
all_forces = data['forces']

# Plotting selected frames
num_frames = all_frames.shape[0]
selected_indices = range(0, num_frames, 100)  # Adjust step for fewer frames

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

# Plotting force data
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

# Setting up the figure and axis for animation with force overlay
fig, ax = plt.subplots()
img = ax.imshow(all_frames[0][:, :, ::-1])  # Display the first frame
ax.axis('off')  # Hide the axes

# Initialize text objects for displaying force data on the animation
text_objects = [ax.text(0.05, 0.05 + 0.05 * i, '', transform=ax.transAxes, color='black', fontsize=12) for i in range(len(force_labels))]

def update(frame_number):
    img.set_data(all_frames[frame_number][:, :, ::-1])  # Update the frame displayed
    artists = [img]  # Start with img as the first artist to update
    for i, txt in enumerate(text_objects):
        txt.set_text(f'{force_labels[i]}: {all_forces[frame_number, i]:.2f}')  # Update force values
        artists.append(txt)  # Add each text artist to the list
    return artists  # Return the list of artists


# Create the animation
ani = FuncAnimation(fig, update, frames=len(all_frames), blit=True, interval=50)  # Adjust interval for frame rate
plt.show()
ani.save('animation_with_forces.gif', writer='ffmpeg', fps=20)
