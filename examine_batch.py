# Script to check the images like a video together with the corresponding
# force measurement in the z direction.

# Import libraries
import os
import numpy as np 
import cv2

# relative path to npz files
path = 'Measurements/attenuation 003'
file_name = 'output_batch_%d.npz'
i = 1 # batch number
file_path = os.path.join(path,file_name %i)
data = np.load(file_path)

cv2.namedWindow('Camera Output', cv2.WINDOW_NORMAL)

# Write some Text
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (30,200)
fontScale              = 1
fontColor              = (255,0,0)
thickness              = 1
lineType               = 2

for frame,forces in zip(data['frames'],data['forces']):
    cv2.putText(frame,"{:.2f}".format(forces[2]), 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    cv2.imshow('Camera Output', frame)
    cv2.waitKey(50) # how many milliseconds to wait between each frame
cv2.destroyAllWindows()