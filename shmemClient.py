import cv2
import numpy as np
from multiprocessing import shared_memory
import time

while True:
    try:
        shm_img = shared_memory.SharedMemory(name="video_frame", create=False, size=1280*720*3)
        break
    except FileNotFoundError:
        time.sleep(0.1)

while True:
    if shm_img is None:
        time.sleep(0.1)
        continue
    # Read the image data from shared memory
    img_data = np.ndarray((720, 1280, 3), dtype=np.uint8, buffer=shm_img.buf)
    
    # Display the image using OpenCV
    cv2.imshow("Shared Memory Video", img_data)
    time.sleep(0.03)  # Add a small delay to control the frame rate
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

shm_img.close()