# Script to check the images like a video together with the corresponding
# force measurement in the z direction.

# Import libraries
import os
import numpy as np 
import cv2

# relative path to npz files
path = ''
path = 'Measurements/21_05_2024 deformed membrane/attenuation 003 random multisine'
file_name = 'output_batch_%d.npz'
i = 5 # batch number
file_path = os.path.join(path,file_name %i)
data = np.load(file_path)
rate = 20

cv2.namedWindow('Unused for paper Output', cv2.WINDOW_NORMAL)

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
    cv2.imshow('Unused for paper Output', frame)
    cv2.waitKey(int(1000*1/rate)) # how many milliseconds to wait between each frame
cv2.destroyAllWindows()
