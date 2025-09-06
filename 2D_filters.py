import cv2
import numpy as np


def local_diff_filter(
        img_gray,
        ksize=(15, 15),
        mode="both",  # both/pos/neg
        border=cv2.BORDER_REFLECT101
):
    gray_f = img_gray.astype(np.float32)

    ref_map = cv2.boxFilter(gray_f, ddepth=-1, ksize=ksize, normalize=True, borderType=border)

    img_diff = gray_f - ref_map

    if mode == "both":
        img_diff = np.abs(img_diff)
    elif mode == "pos":
        pass
    elif mode == 'neg':
        img_diff *= -1

    return np.clip(img_diff, 0, 255).astype(np.uint8)


def variance_filter(
        img_gray,
        ksize=(15, 15),
        gaussian=False,
        border=cv2.BORDER_REFLECT101,
        weight=1.0
):
    gray_f = img_gray.astype(np.float32)

    if gaussian:
        mean_map = cv2.GaussianBlur(gray_f, ksize, 0, borderType=border)
        mean_sq_map = cv2.GaussianBlur(gray_f * gray_f, ksize, 0, borderType=border)
    else:
        mean_map = cv2.boxFilter(gray_f, ddepth=-1, ksize=ksize, borderType=border, normalize=True)
        mean_sq_map = cv2.boxFilter(gray_f * gray_f, ddepth=-1, ksize=ksize, borderType=border, normalize=True)

    # 分散フィルタ
    var_map = mean_sq_map - mean_map * mean_map
    var_map = np.clip(var_map, 0, None)
    std_map = np.sqrt(var_map)

    return np.clip(std_map * weight, 0, 255).astype(np.uint8)
