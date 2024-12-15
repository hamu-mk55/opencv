import os
import random
import functools
import csv

import cv2
import numpy as np


class BlobInfo:
    def __init__(self):
        self.marker_num = None

        # 輪郭情報
        self.x0 = None
        self.y0 = None
        self.w = None
        self.h = None

        self.area = None
        self.mean = None
        self.arc_length = None
        self.xc_ellipse = None
        self.yc_ellipse = None
        self.w_ellipse = None
        self.h_ellipse = None
        self.deg_ellipse = None

        # 距離情報
        self.dist_max = None
        self.xc_dist = None
        self.yc_dist = None

    def correct_coordinate(self, x0, y0):
        self.x0 += x0
        self.y0 += y0

        self.xc_dist += x0
        self.yc_dist += y0

        self.xc_ellipse += x0
        self.yc_ellipse += y0

    @property
    def is_circle(self):
        if self.h_ellipse is None or self.w_ellipse is None:
            return False

        ratio = self.h_ellipse / self.w_ellipse

        if 1 / 1.3 < ratio < 1.3:
            return True
        else:
            return False

    @property
    def headers(self):
        headers = ["marker_num",
                   "x0", "y0", "width", "height",
                   "area", "mean",
                   "ellipse_xc", "ellipse_yc",
                   "ellipse_w", "ellipse_h",
                   "ellipse_degree",
                   "dist"
                   ]

        return headers

    @property
    def vals(self):
        vals = [self.marker_num,
                self.x0, self.y0, self.w, self.h,
                self.area, self.mean,
                self.xc_ellipse, self.yc_ellipse,
                self.w_ellipse, self.h_ellipse,
                self.deg_ellipse,
                self.dist_max]

        return vals


@functools.cache
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


def delete_small_area(img_bin: np.ndarray,
                      min_blob: int = 5,
                      min_hole: int = 5,
                      max_hole: int = 100) -> np.ndarray:
    """
    二値化画像に対し、小さいブロブおよび穴を消す
    :param img_bin:
    :param min_blob:
    :param min_hole:
    :param max_hole:
    :return:
    """

    def _pickup_contours(_img_bin, _min_area, _max_area=None):
        # cv2.RETR_CCOMPで白領域の外側/内側輪郭を取得
        _contours, _hierarchy = cv2.findContours(_img_bin,
                                                 cv2.RETR_CCOMP,
                                                 cv2.CHAIN_APPROX_SIMPLE)

        _contours_list = []
        for _contour, _h_info in zip(_contours, _hierarchy[0]):
            # h_info: [next-no, previous-no, child_no, parent-no]
            # if not exist, contour-no is -1
            _child = _h_info[2]
            _parent = _h_info[3]

            if _parent > -1:
                continue
            if cv2.contourArea(_contour) < _min_area:
                continue
            if _max_area is not None and cv2.contourArea(_contour) > _max_area:
                continue

            _contours_list.append(_contour)

        return _contours_list

    # 白領域の外側輪郭を取得
    contours_outside = _pickup_contours(img_bin, _min_area=min_blob)

    outside = np.zeros_like(img_bin)
    outside = cv2.drawContours(outside, contours_outside, -1, (255, 255, 255), -1)

    # 白領域の内側輪郭を取得
    img_bin = cv2.bitwise_not(img_bin)
    contours_inside = _pickup_contours(img_bin, _min_area=min_hole, _max_area=max_hole)

    inside = np.full_like(img_bin, 255)
    inside = cv2.drawContours(inside, contours_inside, -1, (0, 0, 0), -1)

    img_bin = cv2.bitwise_and(outside, inside)

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


def check_blob_info(img_bin, img_dist):
    blob = BlobInfo()

    # 距離変換画像
    _dist_max = np.max(img_dist)
    _, _, _, max_pos = cv2.minMaxLoc(img_dist)
    blob.dist_max = _dist_max
    blob.xc_dist = max_pos[1]
    blob.yc_dist = max_pos[0]

    # 輪郭情報
    contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    blob.x0, blob.y0, blob.w, blob.h = cv2.boundingRect(contours[0])
    blob.area = cv2.contourArea(contours[0])
    blob.arc_length = cv2.arcLength(contours[0], True)
    _pos, _size, blob.deg_ellipse = cv2.fitEllipse(contours[0])
    blob.xc_ellipse, blob.yc_ellipse = _pos
    blob.h_ellipse, blob.w_ellipse = _size

    return blob


def exec_watershed(img_org: np.ndarray,
                   img_bin: np.ndarray,
                   min_radius: int = 3,
                   use_bin_image: bool = False,
                   draw_sure_fg: bool = True):
    img_seg = np.zeros_like(img_org)
    blob_list = []

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
    kernel = make_kernel(k_size=int(ksize), is_rectangle=False)
    img_dist_max = cv2.dilate(img_dist, kernel, iterations=1)

    # 極大値のリストを取得した後、各領域ごとに前景を取得
    dist_list = np.unique(img_dist_max[img_dist == img_dist_max])
    dist_list = [float(x) for x in dist_list if x > min_radius]

    # 空の場合は終了
    if len(dist_list) == 0:
        return blob_list, img_seg

    for dist in dist_list:
        _fg = np.zeros_like(img_bin)
        _fg[img_dist == dist] = 255
        # たまたま極大値と同じ領域に過検出するのを抑止
        _fg[img_dist != img_dist_max] = 0

        # 極大値の0.5倍分を膨張させ、明確な前景とする
        ksize = (dist * 0.5) * 2 + 1
        kernel = make_kernel(k_size=int(ksize), is_rectangle=False)
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
        _img_dist = img_dist / np.max(img_dist)
        _img_bin = img_bin * _img_dist
        _img_bin = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(_img_bin, markers)

    # 出力画像の設定
    for cnt in range(np.max(markers)):
        cnt += 1

        # skip bg
        if cnt == 1:
            continue

        # ブロブ確認
        _img_dist = img_dist.copy()
        _img_dist[markers != cnt] = 0
        _img_bin = np.zeros_like(img_bin)
        _img_bin[markers == cnt] = 255
        _img_out = img_org[markers == cnt]
        blob = check_blob_info(_img_bin, _img_dist)
        blob.marker_num = np.max(markers) - 1
        blob.mean = np.mean(_img_out)
        blob_list.append(blob)

        # segmentation
        col = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        img_seg[markers == cnt] = col

    if draw_sure_fg:
        img_seg[sure_fg == 255] = (255, 255, 255)

    return blob_list, img_seg


def blob_detection(img_org: np.ndarray,
                   img_bin: np.ndarray,
                   min_length: int = 3,
                   margin: int = 5,
                   csv_file: str | None = None,
                   debug_dir: str | None = None):
    # 事前準備
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
    if csv_file is not None:
        fw = open(csv_file, 'a')
        csvWriter = csv.writer(fw, lineterminator='\n')

    img_seg = np.zeros_like(img_org)

    # ブロブ処理
    label_list, label_img = make_blob_image(img_bin)

    label_cnt = 0
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

        _blob_list, _img_seg = exec_watershed(_img_org, _img_bin, use_bin_image=False)
        img_seg[_y0:_y1, _x0:_x1] += _img_seg

        # 出力
        for blob in _blob_list:
            label_cnt += 1

            # BLOB画像出力
            if debug_dir is not None:
                if blob.is_circle:
                    out_dir = f"{debug_dir}/circle"
                else:
                    out_dir = f"{debug_dir}/ellipse"
                os.makedirs(out_dir, exist_ok=True)

                _img_out = _img_org.copy()
                cv2.rectangle(_img_out,
                              (blob.x0, blob.y0),
                              (blob.x0 + blob.w, blob.y0 + blob.h),
                              color=(255, 0, 0), thickness=2)

                _img_out = cv2.hconcat([_img_out, _img_seg, cv2.cvtColor(_img_bin, cv2.COLOR_GRAY2BGR)])
                _img_out = cv2.resize(_img_out, None, fx=3.0, fy=3.0)
                cv2.imwrite(f"{out_dir}/{label_cnt:04d}.png", _img_out)

            # CSV出力
            if csv_file is not None:
                blob.correct_coordinate(x0=_x0, y0=_y0)
                csvWriter.writerow([label_cnt] + blob.vals)

    if csv_file is not None:
        fw.close()

    return img_seg


def blob_analysis(img_file: str):
    csvfile = "test.csv"
    fw = open(csvfile, 'w')
    csvWriter = csv.writer(fw, lineterminator='\n')
    _blob = BlobInfo()
    csvWriter.writerow(["label_cnt"] + _blob.headers)
    fw.close()

    # 画像読込
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img_out = img.copy()

    # 二値化: グレイスケール->二値化->穴埋めなど
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    img = delete_small_area(img, min_blob=0, min_hole=2)

    # ブロブ処理＋領域分割処理
    img_seg = blob_detection(img_out, img,
                             csv_file=csvfile, debug_dir="./debug")

    cv2.imwrite("test.png", img_seg)


if __name__ == '__main__':
    blob_analysis(img_file="./images2/102.jpg")
