import cv2
import numpy as np

path = "/Users/davinderkumar/Essentials/LOP/img/IX-11-01917_0004_0002.JPG"
img = cv2.imread(path)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
# sift = cv2.SIFT()
kp = sift.detect(img,None)

temp = np.zeros(shape = img.shape)
img=cv2.drawKeypoints(img,kp, outImage=temp)

cv2.imwrite('sift_keypoints.jpg',img)