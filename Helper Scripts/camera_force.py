import threading
import queue
import serial
import time
import re
import struct
import datetime
import pandas as pd
from pypylon import pylon
import cv2
import numpy as np

# Unused for paper and batch processing parameters
batch_size = 1000
target_resolution = (480, 270)
batch_queue = queue.Queue()
recording = True

## Saving thread

# Function to save batch
def save_batch(frames, forces, batch_id):
    np.savez_compressed(f'output_batch_{batch_id}.npz', frames=np.array(frames), forces=np.array(forces))

# Batch saver thread
def batch_saver_thread():
    while True:
        batch_id, frames, forces = batch_queue.get()
        if frames is None:
            batch_queue.task_done()
            break
        save_batch(frames, forces, batch_id)
        batch_queue.task_done()

# Start the batch saver thread
threading.Thread(target=batch_saver_thread, daemon=True).start()

## Keyboard interrupt thread

# Read a keyboard input and set flag to false
def read_keyboard():
    global recording
    input("Press any key to stop recording\n")
    recording = False

threading.Thread(target=read_keyboard, daemon=True).start()

# Unused for paper setup
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

frames = []
forces = []
batch_count = 0

# Enable matrix calculation on electronics 
SAMPLE = 100  # Sample rate as set by DIP configuration 100 Hz, 500Hz, 1000Hz
subsample = 5 # it will take one every x force measurements, dividing the force sensor rate by x
count = 1 # helper variable
PORT = 'COM5'

# Start serial communication
ser = serial.Serial(PORT, 12000000)

start = time.time()
while camera.IsGrabbing() and recording:
    serial_line = ser.read(28)
    if count == 1:
        [F_x, F_y, F_z, M_x, M_y, M_z, temp] = struct.unpack('fffffff', serial_line[0:28])
        current_force = [F_x, F_y, F_z, M_x, M_y, M_z, temp]
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            img = image.GetArray()
            resized_img = cv2.resize(img, target_resolution)
            frames.append(resized_img)
            forces.append(current_force)

            if len(frames) >= batch_size:
                batch_queue.put((batch_count, frames.copy(), forces.copy()))
                frames.clear()
                forces.clear()
                batch_count += 1
                end = time.time()
                print('Elapsed time:',end - start) # the printed time should be batch_size*(1 / (SAMPLE/subsample))
                start = time.time()

        grabResult.Release()
    elif count == subsample:
        count = 0
    count += 1

# Clean up
ser.close()
camera.StopGrabbing()
cv2.destroyAllWindows()

# Finalizing and saving any remaining data
if frames:
    batch_queue.put((batch_count, frames.copy(), forces.copy()))
batch_queue.put((None, None, None))  # End signal
print("waiting for saving thread...")
batch_queue.join()

