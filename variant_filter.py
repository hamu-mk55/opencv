import random

import cv2
import numpy as np


def variant_filter(img, kernel_size):
    img = img.astype(np.float32)

    img_avg = cv2.blur(img, (kernel_size, kernel_size))

    _img = img - img_avg
    _img = _img * _img
    _img = cv2.blur(_img, (kernel_size, kernel_size))

    _img = np.clip(_img, 0, 255).astype(np.uint8)

    return _img


def example(debug=True):
    img_file = 'sample.jpg'
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

    img_out = variant_filter(img, kernel_size=11)

    cv2.imwrite('std.jpg', img_out)


if __name__ == '__main__':
    example()
