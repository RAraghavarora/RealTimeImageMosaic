import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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


def image_split(path, name=''):
    '''
    Split the given image into multiple images with the given overlap
    '''
    img = cv2.imread(path)
    img_h, img_w, _ = img.shape    
    split_width = 4000
    split_height = 4000
    overlap = 0.5  # 50% overlap
    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)
    count = 0
    name = 'images/' + name + '_splitted'
    frmt = 'jpeg'
    out_images = []
    for i in Y_points:
        for j in X_points:
            split = img[i:i + split_height, j:j + split_width]
            out_images.append(split)
            cv2.imwrite('{}_{}.{}'.format(name, count, frmt), split)
            count += 1
    return out_images


# p1 = '/Users/davinderkumar/Essentials/LOP/Codes/images/input1.JPG'
# p2 = '/Users/davinderkumar/Essentials/LOP/Codes/images/input2.JPG'
# p3 = '/Users/davinderkumar/Essentials/LOP/Codes/images/input3.JPG'

# image_split(p1, 'input1')
