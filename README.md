# Vision-Based Soft Sensor
## [Encoder CNN input force.py] and [Encoder CNN input image.py]

### Overview
This Jupyter Notebook is designed for training and evaluating the Convolutional Neural Network (CNN) SUBNET that incorporates input force data. The model processes a dataset consisting of 20 frames and 3 forces, aiming to predict outcomes based on these inputs.

### Purpose
The primary objective of this notebook is to explore the impact of various hyperparameters on the performance of the CNN SUBNET. Different configurations such as the number of hidden states, number of frames, and number of forces are tested to optimize model effectiveness.

### Contents
1. **Training the Model**: The notebook includes code cells that initialize and train the CNN SUBNET using the combined dataset.
2. **Evaluation**: After training, the model is evaluated using a test set to assess its predictive accuracy and other performance metrics.
3. **Results Visualization**: Results from the evaluations are visualized and saved in a designated figures folder. This section helps in understanding the model's performance across different hyperparameters.

### Usage
To use this notebook:
- Ensure all dependencies are installed as specified in the preceding cells.
- Execute the notebook cells sequentially to train and evaluate the model.
- Review the figures in the figures folder to analyze the impact of different configurations.

This tool is particularly useful for researchers and engineers interested in machine learning applications in dynamic environments where force input plays a crucial role.


## Frame and Force Data Visualization Scripts
### examine_batch.py
This script visualizes frames from a `.npz` file as a video with overlaid force measurements in the z direction.

#### Features

- **Load Data:**
  - Loads frames and forces from a specified `.npz` file.

- **Display Frames:**
  - Displays frames in a window with the z-direction force measurement overlaid as text.

#### Usage

1. **Adjust Parameters:**
   - Update `path`, `file_name`, and `i` to point to the correct batch file.
   - Adjust `rate` to change the frame display rate.

2. **Run Script:**
   - Opens a window displaying the frames as a video with the force measurement.

This script provides an easy way to visually inspect frame data alongside corresponding force measurements.

### readoutcs.py

This script visualizes frames and force data from a combined `.npz` file. It plots selected frames, force distributions, and creates an animated video with force data overlay.

#### Features

- **Load Data:**
  - Load frames and forces from `combined_output_20_3_new.npz`.

- **Plot Selected Frames:**
  - Plots a grid of selected frames.

- **Plot Force Data:**
  - Plots force distributions for each force component.

- **Create Animation:**
  - Animates frames with overlaid force data.
  - Saves the animation as `animation_with_forces.gif`.

#### Usage

1. **Adjust Parameters:**
   - Update `filen` if needed.
   - Modify `selected_indices`, `rows`, and `cols` for different frame selections and grid size.
   - Adjust animation `interval` and `fps`.

2. **Run Script:**
   - Displays plots and creates an animated video.

This script helps in visualizing and analyzing the data effectively.

## Real-Time Data Acquisition Script
### camera_force_sensor.py
This script captures and processes real-time data from a camera and force sensor, saving it in batches using multithreading.

#### Features

- **Multithreading:** 
  - **Batch Saver Thread:** Saves data batches asynchronously.
  - **Keyboard Interrupt Thread:** Stops recording on user input.
  
- **Data Acquisition:**
  - Captures images from a camera.
  - Reads force sensor data via serial communication.

- **Batch Processing:** 
  - Collects data in batches of size `batch_size`.
  - Saves each batch as a compressed `.npz` file.

#### Usage

1. **Adjust Parameters:**
   - `batch_size`, `target_resolution`, `PORT`.

2. **Run Script:**
   - Starts capturing and processing data.
   - Stops when a key is pressed.

3. **Output:**
   - Saves batches as `output_batch_{batch_id}.npz`.

This ensures efficient real-time data processing and storage.

## Helper Scripts

### combinev2.py

`combinev2.py` script combines data from multiple `.npz` files into a single compressed `.npz` file. It processes a specified number of batch files, aggregates their contents by key, and saves the combined data efficiently.

- **Adjust the number of files**: Modify the `num_files` variable to match your total batch files.
- **File Paths**: Ensure `first_file_path` and `file_path` are correctly set to your files' locations.
- **Execution**: Run the script to generate a new file `combined_output_20_3_new.npz` with all combined arrays.

Upon successful execution, the script prints the keys of the combined data for verification.

### plot_scripts.py

#### `strip_plotter` Function

This function creates strip plots comparing system output with CNN encoder predictions.

- **Parameters:**
  - `test`, `test_p`: Test and predicted datasets.
  - `norm`: Normalization factors.
  - `to_img`: Function to convert data to images.
  - `plot_image`: Optional custom image plotting function.
  - `semi_log_y`: Whether to use a semi-logarithmic scale for the y-axis.
  - `n_plots`: Number of plots to generate.
  - `off_set`: Offset for the plots.
  - `f_s`: Sampling frequency.
  - `filename`: Name of the output file.
  - `cmap`: Colormap for the images.

- **Usage:** 
  Adjust parameters as needed and run the function to generate a strip plot saved as `.png`.

#### `make_video` Function

This function creates an animated video comparing system output with CNN encoder predictions over time.

- **Parameters:**
  - `test`, `test_p`: Test and predicted datasets.
  - `norm`: Normalization factors.
  - `to_img`: Function to convert data to images.
  - `target_fps`: Target frames per second for the video.
  - `f_s`: Sampling frequency.
  - `filename`: Name of the output video file.
  - `cmap`: Colormap for the images.

- **Usage:** 
  Adjust parameters as needed and run the function to generate a video saved as `.mp4`.

Both functions utilize `matplotlib` for plotting and `tqdm` for progress visualization, ensuring efficient and clear visualization of model performance.

