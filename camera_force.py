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

# Camera and batch processing parameters
batch_size = 1000
target_resolution = (480, 270)
batch_queue = queue.Queue()
force_queue = queue.Queue()

# Function to save batch
def save_batch(frames, forces, batch_id):
    np.savez_compressed(f'output_batch_{batch_id}.npz', frames=np.array(frames), forces=np.array(forces))

# Function to read 6DOF sensor data
def read_force_data():
    ser = serial.Serial('COM6', 12000000)
    while True:
        serial_line = ser.read(28)
        [F_x, F_y, F_z, M_x, M_y, M_z, temp] = struct.unpack('fffffff', serial_line[0:28])
        force_queue.put([F_x, F_y, F_z, M_x, M_y, M_z, temp])
    ser.close()

# Start force data reading thread
threading.Thread(target=read_force_data, daemon=True).start()

# Batch saver thread
def batch_saver_thread():
    while True:
        batch_id, frames, forces = batch_queue.get()
        if frames is None:
            break
        save_batch(frames, forces, batch_id)
        batch_queue.task_done()

# Start the batch saver thread
threading.Thread(target=batch_saver_thread, daemon=True).start()

# Camera setup
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

frames = []
forces = []
batch_count = 0

while camera.IsGrabbing():
    if not force_queue.empty():
        current_force = force_queue.get()

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

        cv2.namedWindow('Camera Output', cv2.WINDOW_NORMAL)
        cv2.imshow('Camera Output', resized_img)
        if cv2.waitKey(1) == 27:
            break

    grabResult.Release()

# Clean up
camera.StopGrabbing()
cv2.destroyAllWindows()

# Finalizing and saving any remaining data
if frames:
    batch_queue.put((batch_count, frames, forces))
batch_queue.put((None, None, None))  # End signal
batch_queue.join()
