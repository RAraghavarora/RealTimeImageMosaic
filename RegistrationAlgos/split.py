import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if size == split_size:
            break
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


def add_noise(img):
    # Generate Gaussian noise
    gauss = np.random.normal(0, 1, img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
    # Add the Gaussian noise to the image
    img_gauss = cv2.add(img, gauss)
    # Display the image
    return img


def create_affine_transform(img):
    rows, cols, ch = img.shape

    pts1 = np.float32([[50, 50], 
                       [200, 50],
                       [50, 200]])

    pts2 = np.float32([[10, 100],
                       [200, 50], 
                       [100, 250]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def image_split(path, name=''):
    '''
    Split the given image into multiple images with the given overlap
    '''
    img = cv2.imread(path)
    img_h, img_w, _ = img.shape 
    print(img.shape)   
    split_width = int(0.7 * img_w)
    split_height = img_h
    overlap = 0.4  # 50% overlap
    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)
    count = 0
    name = 'compare/' + name + '_splitted'
    frmt = 'jpeg'
    out_images = []
    for i in Y_points:
        for j in X_points:
            split = img[i:i + split_height, j:j + split_width]
            out_images.append(split)
            # cv2.imwrite('{}_{}.{}'.format(name, count, frmt), split)
            count += 1

    # Add gaussian noise
    out_images[1] = add_noise(out_images[1])

    # Create affine transformation
    out_images[1] = create_affine_transform(out_images[1])
    return out_images


# p1 = '/Users/davinderkumar/Essentials/LOP/Codes/images/input1.JPG'
# p2 = '/Users/davinderkumar/Essentials/LOP/Codes/images/input2.JPG'
# p3 = '/Users/davinderkumar/Essentials/LOP/Codes/images/input3.JPG'

# image_split(p1, 'input1')
