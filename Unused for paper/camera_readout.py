from pypylon import pylon
import cv2
import numpy as np
import threading
import queue

# Parameters
batch_size = 50
target_resolution = (160, 120)  # Reduced resolution (width, height)
batch_queue = queue.Queue()

def save_batch(frames, batch_id):
    np.savez_compressed(f'output_batch_{batch_id}.npz', frames=np.array(frames))

def batch_saver_thread():
    while True:
        batch_id, frames = batch_queue.get()
        if frames is None:
            break
        save_batch(frames, batch_id)
        batch_queue.task_done()

# Start the batch saver thread
threading.Thread(target=batch_saver_thread, daemon=True).start()

# Connecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# Converting to OpenCV bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

frames = []
batch_count = 0

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()

        # Resize the image
        resized_img = cv2.resize(img, target_resolution)
        frames.append(resized_img)

        if len(frames) >= batch_size:
            batch_queue.put((batch_count, frames.copy()))
            frames.clear()
            batch_count += 1

        # Show image (optional)
        cv2.namedWindow('Unused for paper Output', cv2.WINDOW_NORMAL)
        cv2.imshow('Unused for paper Output', resized_img)
        k = cv2.waitKey(1)
        if k == 27:
            break

    grabResult.Release()

# Releasing the resource
camera.StopGrabbing()
cv2.destroyAllWindows()

# Finalize
if frames:
    batch_queue.put((batch_count, frames))
batch_queue.put((None, None))  # End signal
batch_queue.join()
