# FAST Corner Detector
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/Users/davinderkumar/Essentials/LOP/img/IX-11-01917_0004_0002.JPG', 0)


# Initiate FAST object with default values
fast_nms = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=False)
# fast = cv2.FastFeatureDetector()

# find and draw the keypoints
kp = fast.detect(img, None)
kp_nms = fast_nms.detect(img,None)
temp = np.zeros(shape=img.shape)
img2 = cv2.drawKeypoints(img, kp, temp, color=(255, 0, 0))
img2_nms = cv2.drawKeypoints(img, kp_nms, temp, color=(255, 0, 0))

# # Print all default params
# print("Threshold: ", fast.getInt('threshold'))
# print("nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
# print("neighborhood: ", fast.getInt('type'))
# print("Total Keypoints with nonmaxSuppression: ", len(kp))

cv2.imwrite('fast_nms_true.png', img2_nms)
cv2.imwrite('fast_nms_false.png', img2)

# # Disable nonmaxSuppression
# fast.setNonmaxSupression(0)
# kp = fast.detect(img, None)

# print("Total Keypoints without nonmaxSuppression: ", len(kp))

# img3 = cv2.drawKeypoints(img, kp, color=(255, 0, 0))

# cv2.imwrite('fast_false.png', img3)
