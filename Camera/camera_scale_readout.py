import serial
import time
import re
import threading
import queue
from pypylon import pylon
import cv2
import numpy as np

# Scale parameters
port = 'COM5'
baudrate = 9600
scale_read_interval = 0.025  # Interval for 20 Hz scale reading

# Camera parameters
batch_size = 200
target_resolution = (160, 120)
batch_queue = queue.Queue()

# Queue for scale data (timestamp, weight)
weight_queue = queue.Queue(maxsize=100)  # Adjust size as needed

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
    while True:
        time_stamp = time.time()
        raw_data = read_weight()
        if raw_data:
            weight = extract_weight(raw_data)
            if weight:
                weight_queue.put((time_stamp, weight))
        time.sleep(scale_read_interval)

# Start the weight reading thread
threading.Thread(target=weight_reading_thread, daemon=True).start()

def save_batch(frames, weights, batch_id):
    np.savez_compressed(f'output_batch_{batch_id}.npz', frames=np.array(frames), weights=np.array(weights))

def batch_saver_thread():
    while True:
        batch_id, frames, weights = batch_queue.get()
        if frames is None:
            break
        save_batch(frames, weights, batch_id)
        batch_queue.task_done()

threading.Thread(target=batch_saver_thread, daemon=True).start()

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

frames = []
weights = []
batch_count = 0

while camera.IsGrabbing():
    frame_start_time = time.time()
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        img = image.GetArray()
        resized_img = cv2.resize(img, target_resolution)
        frames.append(resized_img)

        # Associate scale reading with the frame
        while not weight_queue.empty():
            weight_time_stamp, weight = weight_queue.queue[0]  # Peek the next item
            if weight_time_stamp <= frame_start_time:
                weight_queue.get()  # Remove the item from the queue
                weights.append(weight)
            else:
                break

        if len(frames) >= batch_size:
            batch_queue.put((batch_count, frames.copy(), weights.copy()))
            frames.clear()
            weights.clear()
            batch_count += 1

        cv2.namedWindow('Camera Output', cv2.WINDOW_NORMAL)
        cv2.imshow('Camera Output', resized_img)
        if cv2.waitKey(1) == 27:  # Exit on pressing 'Esc'
            break

    grabResult.Release()

camera.StopGrabbing()
cv2.destroyAllWindows()
if frames:
    batch_queue.put((batch_count, frames, weights))
batch_queue.put((None, None, None))  # End signal
batch_queue.join()
