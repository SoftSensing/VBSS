import threading
import queue
import serial
import time
import re
from pypylon import pylon
import cv2
import numpy as np

# Scale parameters
port = 'COM5'
baudrate = 9600
scale_read_interval = 0.05  # Interval for scale readings (20 Hz)
weight_queue = queue.Queue()

def read_weight():
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            data = ser.readline().decode('ascii').strip()
            return data
    except serial.SerialException as e:
        print(f"Error: {e}")
        return None

def extract_weight(data):
    match = re.search(r'(\d+\.\d+)', data)
    if match:
        return match.group(1)
    else:
        return None

def weight_reading_thread():
    last_read_time = 0
    while True:
        current_time = time.time()
        if current_time - last_read_time >= scale_read_interval:
            last_read_time = current_time
            raw_data = read_weight()
            if raw_data:
                weight = extract_weight(raw_data)
                if weight:
                    weight_queue.put((last_read_time, weight))
        time.sleep(0.001)  # Short sleep to avoid high CPU usage

# Start the weight reading thread
threading.Thread(target=weight_reading_thread, daemon=True).start()

# Unused for paper parameters
batch_size = 50
target_resolution = (160, 120)
batch_queue = queue.Queue()

def save_batch(frames, weights, batch_id):
    np.savez_compressed(f'output_batch_{batch_id}.npz', frames=np.array(frames), weights=np.array(weights))

def batch_saver_thread():
    while True:
        batch_id, frames, weights = batch_queue.get()
        if frames is None:
            break
        save_batch(frames, weights, batch_id)
        batch_queue.task_done()

# Start the batch saver thread
threading.Thread(target=batch_saver_thread, daemon=True).start()

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

frames = []
weights = []
frame_time_stamps = []
batch_count = 0

while camera.IsGrabbing():
    frame_start_time = time.time()

    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        img = image.GetArray()
        resized_img = cv2.resize(img, target_resolution)
        frames.append(resized_img)
        frame_time_stamps.append(frame_start_time)

        # Synchronize weights with frames
        while not weight_queue.empty():
            weight_time_stamp, weight = weight_queue.queue[0]  # Peek at the next item
            if weight_time_stamp <= frame_start_time:
                _, weight = weight_queue.get()  # Get the actual weight
                weights.append(weight)
            else:
                weights.append(None)  # Append None if no matching weight
                break

        if len(frames) >= batch_size:
            batch_queue.put((batch_count, frames.copy(), weights.copy()))
            frames.clear()
            weights.clear()
            frame_time_stamps.clear()
            batch_count += 1

        cv2.namedWindow('Unused for paper Output', cv2.WINDOW_NORMAL)
        cv2.imshow('Unused for paper Output', resized_img)
        if cv2.waitKey(1) == 27:  # Exit on pressing 'Esc'
            break

    grabResult.Release()

camera.StopGrabbing()
cv2.destroyAllWindows()

# Finalizing and saving any remaining data
if frames:
    batch_queue.put((batch_count, frames, weights))
batch_queue.put((None, None, None))  # End signal
batch_queue.join()
