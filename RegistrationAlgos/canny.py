import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
path = "/Users/davinderkumar/DIP/dip_image.jpg"
img = cv.imread(path,0)
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()