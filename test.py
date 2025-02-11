import cv2
import numpy as np


def _make_dummy_img(dx=0, dy=0):
    img_size = 50000
    circle_size = img_size/5
    x0 = int(img_size / 2 + dx)
    y0 = int(img_size / 2 + dy)
    fg = 200
    bg = 50
    gauss = 2

    img = np.full((img_size,img_size),bg, dtype=np.uint8)

    cv2.circle(img, (x0, y0), int(circle_size/2), fg, thickness=-1)

    img = cv2.resize(img, None, fx=0.02, fy=0.02)

    img = cv2.GaussianBlur(img, (7, 7), sigmaX=0)

    noise = np.random.randint(0, 10, (img.shape[0], img.shape[1]), dtype=np.uint8)
    img += noise

    return img


if __name__ == '__main__':
    out_dir = './images3'

    img = _make_dummy_img(dx=0, dy=0)
    cv2.imwrite("org.jpg", img)

    cv2.imwrite("template.jpg", img[300:700,300:700])

    for x_cnt in range(200):
        dx = x_cnt

        img = _make_dummy_img(dx=dx, dy=0)

        cv2.imwrite(f"{out_dir}/x{x_cnt:03d}.jpg", img)
