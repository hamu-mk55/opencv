import os
import random

import cv2
import numpy as np


def delete_small_area(img_bin: np.ndarray,
                      min_blob: int = 5, min_hole: int = 5) -> np.ndarray:
    """
    二値化画像に対し、小さいブロブおよび穴を消す
    :param img_bin:
    :param min_blob:
    :param min_hole:
    :return:
    """

    # cv2.RETR_CCOMPで白領域の外側/内側輪郭を取得
    contours, hierarchy = cv2.findContours(img_bin,
                                           cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)

    contours_outside = []
    contours_inside = []
    for contour, h_info in zip(contours, hierarchy[0]):
        # h_info: [next-no, previous-no, child_no, parent-no]
        # if not exist, contour-no is -1
        _child = h_info[2]
        _parent = h_info[3]

        if _parent > -1:
            # inside-contour
            if cv2.contourArea(contour) >= min_hole:
                contours_inside.append(contour)
        else:
            # outside
            if cv2.contourArea(contour) >= min_blob:
                contours_outside.append(contour)

    img_bin = cv2.drawContours(img_bin, contours_outside, -1, (255, 255, 255), -1)
    img_bin = cv2.drawContours(img_bin, contours_inside, -1, (0, 0, 0), -1)

    return img_bin


def make_blob_image(img_bin: np.ndarray) -> (list, np.ndarray):
    """
    ブロブ処理
    :param img_bin:
    :return:
    """
    # notes: label_num involve background-label(0)
    # notes: label_data=[[x0,y0,width,height,area]]
    # notes: label_list=[[label_no,x0,y0,width,height,area]]
    label_num, label_img, label_data, _ = cv2.connectedComponentsWithStats(img_bin)
    label_num = label_num - 1
    label_data = np.delete(label_data, 0, 0)

    label_list = []
    for label_cnt in range(label_num):
        _data = list(label_data[label_cnt])

        label_list.append([label_cnt + 1] + _data)

    return label_list, label_img


def make_kernel(k_size: int, is_rectangle: bool = True) -> np.ndarray:
    k_size = int(k_size)
    if k_size % 2 == 0:
        k_size += 1

    kernel = np.ones((k_size, k_size), np.uint8)

    if not is_rectangle:
        xc = (k_size - 1) / 2
        yc = (k_size - 1) / 2
        for x in range(k_size):
            for y in range(k_size):
                radius = ((x - xc) ** 2 + (y - yc) ** 2) ** 0.5

                if radius > (k_size - 1) / 2.0:
                    kernel[y, x] = 0

    return kernel


def exec_watershed(img_org: np.ndarray, img_bin: np.ndarray,
                   min_radius: int = 3,
                   use_bin_image: bool = True):
    img_out = img_org.copy()

    # STEP1: 明確な背景を取得
    kernel = make_kernel(k_size=3, is_rectangle=False)
    sure_bg = cv2.dilate(img_bin, kernel, iterations=3)

    # STEP2:　明確な前景を取得
    # 対象ブロブのサイズは一定ではないため、距離の極大点を求め、そこを明確な前景とする
    sure_fg = np.zeros_like(img_bin)

    # 距離変換
    img_dist = cv2.distanceTransform(img_bin, cv2.DIST_L2, 5)

    # 距離画像を膨張させることで極大領域を見つける
    # くっついている場合、min_radiusの1.5倍までは分離できるようにする
    ksize = (min_radius * 1.5) * 2 + 1
    kernel = make_kernel(k_size=ksize, is_rectangle=False)
    img_dist_max = cv2.dilate(img_dist, kernel, iterations=1)

    # 極大値のリストを取得した後、各領域ごとに前景を取得
    dist_list = np.unique(img_dist_max[img_dist == img_dist_max])
    dist_list = [float(x) for x in dist_list if x > min_radius]
    for dist in dist_list:
        _fg = np.zeros_like(img_bin)
        _fg[img_dist == dist] = 255
        # たまたま極大値と同じ領域に過検出するのを抑止
        _fg[img_dist != img_dist_max] = 0

        # TODO: 極大値の情報取得
        _, _, _, max_loc = cv2.minMaxLoc(_fg)

        # 極大値の0.5倍分を膨張させ、明確な前景とする
        ksize = (dist * 0.5) * 2 + 1
        kernel = make_kernel(k_size=ksize, is_rectangle=False)
        _fg = cv2.dilate(_fg, kernel, iterations=1)

        sure_fg[_fg == 255] = 255

    # STEP3: watershed実行
    # ボーダー領域を取得
    boader = cv2.subtract(sure_bg, sure_fg)

    # 前景ごとにラベル番号を付与した後、ボーダー領域を0とする
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[boader == 255] = 0

    # Watershed適用
    if use_bin_image:
        _img_bin = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(_img_bin, markers)
    else:
        markers = cv2.watershed(img_out, markers)

    # 出力画像の設定
    # 検証用に多めに出力。本番運用では必要最低限に絞り込み
    img_out[markers == -1] = (255, 255, 255)
    img_bin[markers == -1] = 0

    img_dist = np.clip(img_dist, 0, 255).astype(np.uint8)
    img_dist = cv2.cvtColor(img_dist * 10, cv2.COLOR_GRAY2BGR)

    img_seg = np.zeros_like(img_out)
    for cnt in range(np.max(markers)):
        col = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_seg[markers == cnt + 1] = col
    img_seg[sure_fg == 255] = (255, 255, 255)

    return img_out, img_seg, img_dist, img_bin


def _blob_analysis(img_org: np.ndarray, img_bin: np.ndarray,
                   min_length: int = 3,
                   margin: int = 5,
                   debug_dir: str = "./debug"):
    os.makedirs(debug_dir, exist_ok=True)

    label_list, label_img = make_blob_image(img_bin)

    label_cnt = 0
    img_bin = np.zeros_like(img_bin)
    for label in label_list:
        label_no, x0, y0, width, height, _ = label

        # 小さい領域はSKIP
        if width < min_length or height < min_length:
            continue

        # 輪郭処理のため、外周マージン設定。マージン無い場合はSKIP
        _x1 = x0 + width + margin
        _y1 = y0 + height + margin
        _x0 = x0 - margin
        _y0 = y0 - margin

        if _x0 <= 0 or _y0 <= 0 or _x1 >= img_org.shape[1] or _y1 >= img_org.shape[0]:
            continue

        # 領域分割処理
        _img_org = img_org[_y0:_y1, _x0:_x1]
        _img_bin = label_img[_y0:_y1, _x0:_x1]
        _img_bin = cv2.inRange(_img_bin, label_no, label_no)

        _img_out, _img_seg, _img_dist, _img_bin = exec_watershed(_img_org, _img_bin)
        label_cnt += 1

        # 出力
        _img_out = cv2.hconcat([_img_out, _img_dist, _img_seg, cv2.cvtColor(_img_bin, cv2.COLOR_GRAY2BGR)])
        _img_out = cv2.resize(_img_out, None, fx=3.0, fy=3.0)
        cv2.imwrite(f"{debug_dir}/{label_cnt:03d}.png", _img_out)

        _img = img_bin[y0:y0 + height, x0:x0 + width] + _img_bin[margin:margin + height, margin:margin + width]
        img_bin[y0:y0 + height, x0:x0 + width] = _img

    return img_bin


def blob_analysis(img_file: str):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img_out = img.copy()

    # 二値化: グレイスケール->二値化->穴埋めなど
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    cv2.imwrite("test_gray.png", img)

    img = delete_small_area(img, min_blob=5, min_hole=20)

    cv2.imwrite("test_gray2.png", img)

    #
    img_bin = _blob_analysis(img_out, img)

    cv2.imwrite("test.png", img_bin)


if __name__ == '__main__':
    blob_analysis(img_file="./images2/102.jpg")
