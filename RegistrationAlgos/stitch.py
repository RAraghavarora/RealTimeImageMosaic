import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

image1 = cv2.imread('/Users/davinderkumar/Essentials/LOP/img/IX-11-01917_0004_0231.JPG', 0)
image2 = cv2.imread('/Users/davinderkumar/Essentials/LOP/img/IX-11-01917_0004_0232.JPG', 0)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)
print("1")

match = cv2.BFMatcher()
matches = match.knnMatch(des1, des2, k=2)

print("matched")

good = []
for m, n in matches:
    if m.distance < 0.2 * n.distance:
        good.append(m)

draw_params = dict(matchColor = (0, 255, 0),  # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
img3 = cv2.drawMatches(image1, kp1, image2, kp2, good, None, **draw_params)
imgplot = plt.imshow(img3)
plt.show()
# cv2.imshow("original_image_drawMatches.jpg", img3)
# cv2.waitKey(0)

# # Find the Homography transformation

# MIN_MATCH_COUNT = 10
# if len(good) > MIN_MATCH_COUNT:
#     src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)    
#     h, w = image1.shape
#     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
#     dst = cv2.perspectiveTransform(pts, M)    
#     image2 = cv2.polylines(image2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
#     cv2.imshow("original_image_overlapping.jpg", image2)
#     cv2.waitKey(0)
# else:
#     print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

# dst = cv2.warpPerspective(image1, M, (image2.shape[1] + image1.shape[1], image2.shape[0]))
# dst[0:image2.shape[0], 0:image2.shape[1]] = image2
# cv2.imshow("original_image_stitched.jpg", dst)
# cv2.waitKey(0)


def trim(frame):
    # crop top
    if not np.sum(frame[0]):
        print("Crop Top")
        return trim(frame[1:])
    # crop bottom
    if not np.sum(frame[-1]):
        print("Crop Bottom")
        return trim(frame[:-2])
    # crop left
    if not np.sum(frame[:, 0]):
        print("Crop Left")
        return trim(frame[:, 1:])
    # crop right
    if not np.sum(frame[:, -1]):
        try:
            return trim(frame[:, :-2])
        except RecursionError:
            print("Maximum number of recursions reached")
    return frame


# cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
#cv2.imsave("original_image_stitched_crop.jpg", trim(dst))


def image_show(image, title = "Image"):
    cv2.imshow('GoldenGate', gray_img)
    while True:
        k = cv2.waitKey(0) & 0xFF     
        if k == 27:
            break             # ESC key to exit
    cv2.destroyAllWindows()
