import cv2
import numpy as np
import time
from split import start_points, image_split

# image1 = cv2.imread('/Users/davinderkumar/Essentials/LOP/img/IX-11-01917_0004_0002.JPG', 0)
# image2 = cv2.imread('/Users/davinderkumar/Essentials/LOP/img/IX-11-01917_0004_0003.JPG', 0)
path = '/Users/davinderkumar/Essentials/LOP/img/IX-11-01917_0004_00'
path2 = '/Users/davinderkumar/Essentials/LOP/img/IX-11-01917_0004_02'
p = "/Users/davinderkumar/Desktop/images/P0"
im1 = '/Users/davinderkumar/Desktop/images/P0010.png'
im2 = '/Users/davinderkumar/Desktop/images/P0011.png'
img = "/Users/davinderkumar/Desktop/images/P0831.png"


def sift(image1, image2):
    sift = cv2.xfeatures2d.SIFT_create()
    # Get the keypoints and the descriptors
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    return [kp1, des1, kp2, des2]


def surf(image1, image2):
    surf = cv2.xfeatures2d.SURF_create(400)
    kp1, des1 = surf.detectAndCompute(image1, None)
    kp2, des2 = surf.detectAndCompute(image2, None)
    return kp1, des1, kp2, des2


def stitch(image1, image2, function, _):
    kp1, des1, kp2, des2 = function(image1, image2)
    print("Keypoints detected")

    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)
    print("Keypoints matched")

    good = []
    for m, n in matches:
        if m.distance < 0.002 * n.distance:
            good.append(m)

    print(len(good))

    draw_params = dict(matchColor = (0, 155, 0),  # draw matches in green color
                       singlePointColor = None,
                       flags = 2)
    img3 = cv2.drawMatches(image1, kp1, image2, kp2, good, None, **draw_params)

    # cv2.imshow("original_image_drawMatches.jpg", img3)
    # cv2.waitKey(0)
    if function == surf:
        cv2.imwrite('compare/' + str(_) + "_surf.png", img3)

    if function == sift:
        cv2.imwrite('compare/' + str(_) + "_sift.png", img3)

    # Find the Homography transformation

    # MIN_MATCH_COUNT = 10
    # if len(good) > MIN_MATCH_COUNT:
    #     src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    #     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) 

    #     if len(image1.shape) > 2:
    #         h, w, _ = image1.shape
    #     else:
    #         h, w = image1.shape          
    #     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    #     dst = cv2.perspectiveTransform(pts, M)    
    #     image2 = cv2.polylines(image2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    #     # cv2.imshow("original_image_overlapping.jpg", image2)
    #     # cv2.waitKey(0)
    # else:
    #     print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

    # print("Homography computed")
    # dst = cv2.warpPerspective(image1, M, (image2.shape[1] + image1.shape[1], image2.shape[0]))

    # dst[0:image2.shape[0], 0:image2.shape[1]] = image2

    # # # cv2.imshow("original_image_stitched.jpg", dst)
    # # # cv2.waitKey(0)
    # return dst


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


def image_show(image, title = "Image"):
    cv2.imshow(title, image)
    while True:
        k = cv2.waitKey(0) & 0xFF     
        if k == 27:
            break             # ESC key to exit
    cv2.destroyAllWindows()


def image_compare():
    t1 = time.time()
    im1 = cv2.imread(im1)
    im2 = cv2.imread(im2)
    im2 = cv2.rotate(im2, cv2.ROTATE_90_CLOCKWISE) 
    dst = stitch(im1, im2, sift)

    for _ in range(10):
        dst = trim(dst)

    print("Time = ", time.time() - t1)  
    x = cv2.imwrite("Test__sift_output.jpg", dst)


# t1 = time.time()
# image_path = img
# split_imgs = image_split(img, "img")

# dst = stitch(split_imgs[0], split_imgs[1], surf)
# for _ in range(10):
#     dst = trim(dst)
# print(" surf")
# print("Time = ", time.time() - t1)  
# x = cv2.imwrite("./compare/1_surf_output.jpg", dst)

# t1 = time.time()
# dst = stitch(split_imgs[0], split_imgs[1], sift)
# for _ in range(10):
#     dst = trim(dst)
# print(" sift")
# print("Time = ", time.time() - t1)  
# x = cv2.imwrite("./compare/1_sift_output.jpg", dst)


# for i in [1, 2, 3]:
#     t1 = time.time()
#     image_path = p + str(i) + '.JPG'

#     # Get input images from the given image with 50% overlap
#     split_imgs = image_split(image_path, str(i))

#     dst = stitch(split_imgs[0], split_imgs[1], surf)
#     for _ in range(10):
#         dst = trim(dst)
#     print(str(i) + " surf")
#     print("Time = ", time.time() - t1)  
#     x = cv2.imwrite(str(i) + "_surf_output.jpg", dst)

#     dst = stitch(split_imgs[0], split_imgs[1], sift)
#     for _ in range(10):
#         dst = trim(dst)
#     print(str(i) + " surf")
#     print("Time = ", time.time() - t1)  
#     x = cv2.imwrite(str(i) + "_sift_output.jpg", dst)

#     print(x)


for _ in range(0, 907):

    if(_ <= 9):
        p1 = p + '00' + str(_) + '.png'
    elif(_ <= 99):
        p1 = p + '0' + str(_) + '.png'
    else:
        p1 = p + str(_) + '.png'

    try:
        im = cv2.imread(p1)
        if not im.any():
            continue
    except:
        continue

    try:
        split_imgs = image_split(p1, str(_))
    except:
        continue

    t1 = time.time()
    dst = stitch(split_imgs[0], split_imgs[1], surf, _)


# for _ in range(50, 100):
#     t1 = time.time()
#     folder = 'sift/'
#     if(_ <= 9):
#         p1 = path2 + '0' + str(_) + '.JPG'
#     else:
#         p1 = path2 + str(_) + '.JPG'
#     if(_ + 1 <= 9):
#         p2 = path2 + '0' + str(_ + 1) + '.JPG'
#     else:
#         p2 = path2 + str(_ + 1) + '.JPG'

#     image1 = cv2.imread(p1)
#     image2 = cv2.imread(p2)
#     print(p1)
#     print(p2)

#     dst = stitch(image1, image2, sift)
#     for i in range(10):
#         dst = trim(dst)
#     print(str(_) + " sift")
#     print("Time = ", time.time() - t1)  
#     x = cv2.imwrite(str(_) + "co_RA_sift_output.jpg", trim(dst)) 

#     dst = stitch(image1, image2, surf)
#     for i in range(10):
#         dst = trim(dst)
#     print(str(_) + " surf")
#     print("Time = ", time.time() - t1)  
#     x = cv2.imwrite(str(_) + "co_RA_surf_output.jpg", trim(dst)) 
#     print(x)

#     del image1, image2, p1, p2, t1
#     break


# cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
# #cv2.imsave("original_image_stitched_crop.jpg", trim(dst))
