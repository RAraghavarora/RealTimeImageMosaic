import numpy as np
import cv2 as cv
filename = '/Users/davinderkumar/DIP/dip_image2.png'
img = cv.imread(filename,cv.IMREAD_GRAYSCALE)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
# cv.imshow('dst',img)
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()