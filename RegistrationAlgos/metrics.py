import numpy as np
from scipy import ndimage
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import warnings
from math import log10, sqrt
import sklearn.metrics
warnings.filterwarnings('error')

p = "/Volumes/Seagate Portable/LOP/Data/images/P0"
output_file = './output.txt'


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    imageB = cv2.resize(imageB, dsize=imageA.shape[::-1][1:])
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    print(imageA.shape[::-1])
    imageB = cv2.resize(imageB, dsize=imageA.shape[::-1][1:])
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB, multichannel=True)
    return s
    print(s)
    # setup the figure
    # fig = plt.figure(title)
    # plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # # show first image
    # ax = fig.add_subplot(1, 2, 1)
    # plt.imshow(imageA, cmap = plt.cm.gray)
    # plt.axis("off")
    # # show the second image
    # ax = fig.add_subplot(1, 2, 2)
    # plt.imshow(imageB, cmap = plt.cm.gray)
    # plt.axis("off")
    # # show the images
    # plt.show()


def PSNR(image1, image2): 
    # PSNR = 10 log10(R^2/MSE)
    # PSNR = 20 log10(R/MSE)
    image2 = cv2.resize(image2, dsize=image1.shape[::-1][1:])
    mse = np.mean((image1 - image2) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
        # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 


def mutual_information(image1, image2):

    size1 = image1.ravel().shape[0]
    size2 = image2.ravel().shape[0]
    size = min(size1, size2)
    return sklearn.metrics.mutual_info_score(image1.ravel()[:size], image2.ravel()[:size])


# image1 = cv2.imread('/Users/davinderkumar/Essentials/LOP/Codes/RegistrationAlgos/compare/img1.jpg')
# image2 = cv2.imread('/Users/davinderkumar/Essentials/LOP/Codes/RegistrationAlgos/compare/img2.jpg')
# image3 = cv2.imread('./Codes/RegistrationAlgos/final.jpg')

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
    except Exception as e: 
        print(e)
        continue

    ground_truth = im
    surf_path = "/Users/davinderkumar/Essentials/LOP/Codes/RegistrationAlgos/compare/{}_surf_output.jpg".format(_)
    surf = cv2.imread(surf_path)
    sift_path = "/Users/davinderkumar/Essentials/LOP/Codes/RegistrationAlgos/compare/{}_sift_output.jpg".format(_)
    sift = cv2.imread(sift_path)
    ssim_surf = compare_images(surf, ground_truth, "SURF")
    ssim_sift = compare_images(surf, ground_truth, "SIFT")
    mi_surf = mutual_information(ground_truth, surf)
    mi_sift = mutual_information(ground_truth, sift)
    psnr_surf = PSNR(ground_truth, surf)
    psnr_sift = PSNR(ground_truth, sift)
    output = f"{ssim_surf},{mi_surf},{psnr_surf}\n{ssim_sift},{mi_sift},{psnr_sift}\n"
    with open(output_file, 'a') as op:
        op.write(output)
    print(_)


# print(PSNR(ground_truth, sift))

# surf = cv2.imread('/Users/davinderkumar/Essentials/LOP/Codes/50co_RA_surf_output.jpg')
# sift = cv2.imread('/Users/davinderkumar/Essentials/LOP/Codes/50co_RA_sift_output.jpg')

# im1 = cv2.imread('../img/IX-11-01917_0004_0250.JPG')
# im2 = cv2.imread('../img/IX-11-01917_0004_0251.JPG')

# compare_images(im1, surf, 'surf')
# compare_images(im2, surf, 'sift')

# print(mutual_information(im1, surf))
# print(mutual_information(im2, surf))

# print(mse(im1, surf))
# print(mse(im2, surf))

# print(PSNR(im1, surf))
# print(PSNR(im2, surf))
