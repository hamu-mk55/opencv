import random

import cv2
import numpy as np


class DefectInfo:
    def __init__(self):
        self.defect_no = None
        self.group_no = None

        self.x = None
        self.y = None
        self.w = None
        self.h = None

        self.area = None
        self.area_contour = None

        self.mean = None
        self.min = None
        self.max = None

        self.arc_length = None

    def __call__(self):
        return [self.defect_no, self.group_no]

    def input_positions(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def _make_binary_image(img_in,
                       mask_low=None, mask_high=None,
                       mode='hsv',
                       inverse_mask=False,
                       kernel_size=3,
                       erode_iter=0,
                       dilate_iter=0,
                       open_flg=True,
                       mask_img=None):
    # Check parameters......
    # caution: Hue-range is [0:180] in open-cv, [0:255] in image-J
    if mask_low is None:
        raise ValueError('blob_analysis: mask-low is None')
    if mask_high is None:
        raise ValueError('blob_analysis: mask-high is None')

    if mode == 'hsv':
        if img_in.ndim != 3:
            raise ValueError(f'blob_analysis: img-ndim is illegal: {img_in.ndim}')

        img = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    elif mode == 'bgr':
        if img_in.ndim != 3:
            raise ValueError(f'blob_analysis: img-ndim is illegal: {img_in.ndim}')

        img = img_in.copy()
    elif mode == 'gray':
        if img_in.ndim == 3:
            img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
        else:
            img = img_in.copy()
    else:
        raise ValueError(f'blob_analysis: illegal-mode: {mode}')

    # Threshold..........
    img_bin = cv2.inRange(img, mask_low, mask_high)

    if inverse_mask:
        img_bin = cv2.bitwise_not(img_bin, img_bin)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if open_flg:
        img_bin = cv2.erode(img_bin, kernel, iterations=erode_iter)
        img_bin = cv2.dilate(img_bin, kernel, iterations=dilate_iter)
    else:
        img_bin = cv2.dilate(img_bin, kernel, iterations=dilate_iter)
        img_bin = cv2.erode(img_bin, kernel, iterations=erode_iter)

    if mask_img is not None:
        img_bin = cv2.bitwise_and(img_bin, mask_img)

    return img, img_bin


def _make_colors(color_num):
    cols = []
    for cnt in range(color_num + 10):
        cols.append(np.array([random.randint(100, 255),
                              random.randint(100, 255),
                              random.randint(100, 255)]))
    return cols


def _initailize_outputs(img_in, img, label_num, output_seg, output_rec):
    img_bin = np.zeros(img.shape[:2]).astype(np.uint8)
    img_seg = None
    img_rec = None
    cols = None
    if output_seg:
        cols = _make_colors(label_num)
        img_seg = np.zeros(img.shape).astype(np.uint8)
    if output_rec:
        img_rec = img_in.copy()

    return cols, img_bin, img_seg, img_rec


def _make_blob_image(img_bin)->(list, np.array):
    # notes: label[0] involve background-label(0)
    # notes: data_label=[[x0,y0,width,height,area]]
    # notes: label_list=[[label_no,x0,y0,width,height,area]]
    label = cv2.connectedComponentsWithStats(img_bin)
    label_num = label[0] - 1
    label_img = label[1]
    _data_label = np.delete(label[2], 0, 0)

    label_list = []
    for label_cnt in range(label_num):
        _data = list(_data_label[label_cnt])

        label_list.append([label_cnt + 1] + _data)

    return label_list, label_img


def _delete_small_blob_from_bin_image(label_list, label_img, area_min):
    img_bin = np.zeros(label_img.shape[:2]).astype(np.uint8)
    img_bin[label_img > 0] = 255

    if area_min <= 0:
        return img_bin

    for label_data in label_list:
        label_no = label_data[0]
        area_label = label_data[5]

        if area_label > area_min:
            continue

        img_bin[label_img == label_no] = 0

    return img_bin


def _make_group_image(label_list, label_img, group_pixel=3, group_min=0):
    img_bin = _delete_small_blob_from_bin_image(label_list, label_img, group_min)

    _kernel = np.ones((3, 3), np.uint8)
    img_bin = cv2.dilate(img_bin, _kernel, iterations=group_pixel)

    return img_bin


def _check_blob(img, label_img, label_data, img_out, rec_rotate):
    defect = DefectInfo()

    # notes: cv2.inRange is faster than np.where
    label_no = label_data[0]
    _img = cv2.inRange(label_img, label_no, label_no)

    # notes: ret is 2 in opencv4 (3 in opencv3)
    contours, _ = cv2.findContours(_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # search outside-contours(=contour area is maximum)
    max_contour = None
    max_area = 0
    for cont_cnt, contour in enumerate(contours):
        _area_contour = cv2.contourArea(contour)

        if _area_contour > max_area:
            max_area = _area_contour
            max_contour = contour

    # features: intensity
    roi = img[_img > 0]

    if img.ndim == 3:
        defect.mean = (np.mean(roi[:, 0]), np.mean(roi[:, 1]), np.mean(roi[:, 2]))
        defect.max = (np.max(roi[:, 0]), np.max(roi[:, 1]), np.max(roi[:, 2]))
        defect.min = (np.min(roi[:, 0]), np.min(roi[:, 1]), np.min(roi[:, 2]))
    else:
        defect.mean = np.mean(roi)
        defect.max = np.max(roi)
        defect.min = np.min(roi)

    # features: shape
    def _f(input_num):
        return round(input_num, 1)

    _length = cv2.arcLength(max_contour, True)
    defect.arc_length = _f(_length)

    # Min-Rectangle
    if not rec_rotate:
        x, y, w, h = label_data[1:5]

        if img_out is not None:
            cv2.rectangle(img_out, (x, y), (x + w, y + h),
                          color=0, thickness=3)
            cv2.rectangle(img_out, (x, y), (x + w, y + h),
                          color=255, thickness=2)
    else:
        # notes: rect = [(x, y), (w, h), angle]
        _rect = cv2.minAreaRect(max_contour)
        _box = cv2.boxPoints(_rect)
        _box = np.int0(_box)

        x = _f(_rect[0][0])
        y = _f(_rect[0][1])
        h = _f(max(_rect[1][0], _rect[1][1]))
        w = _f(min(_rect[1][0], _rect[1][1]))

        if img_out is not None:
            img_out = cv2.drawContours(img_out, [_box], 0,
                                       color=0, thickness=3)
            img_out = cv2.drawContours(img_out, [_box], 0,
                                       color=255, thickness=2)

    defect.input_positions(x=x, y=y, w=w, h=h)
    defect.area_contour = max_area

    return defect, img_out


def blob_analysis(img_in,
                  mask_low=None, mask_high=None,
                  mode='hsv',
                  inverse_mask=False,
                  kernel_size=3,
                  erode_iter=0, dilate_iter=0,
                  open_flg=True,
                  blob_min=10, blob_max=None,
                  group_pixel=10, group_min=10,
                  mask_img=None,
                  return_if_exist=False,
                  output_seg=False,segment_by_group=True,
                  output_rec=False, rec_rotate=False):
    """
    ブロブ解析を実行
    :param img_in: 入力画像データ(BGR形式 or モノクロ画像)
    :param mask_low: 下側閾値(np.array([x,y,z]) or スカラー)
    :param mask_high: 上側閾値(np.array([x,y,z]) or スカラー)
    :param mode: 二値化の色モード(hsv or bgr or gray)
    :param inverse_mask: 二値化で反転するかどうか
    :param kernel_size: 膨張収縮のカーネルサイズ
    :param erode_iter: 収縮回数
    :param dilate_iter: 膨張回数
    :param open_flg: 収縮->膨張の順にするかどうか
    :param blob_min: ブロブ面積の最小値(Noneの場合指定なし)
    :param blob_max: ブロブ面積の最大値(Noneの場合指定なし)
    :param mask_img: マスク画像
    :param return_if_exist: １個でもブロブが存在したら終了する
    :param output_seg: セグメンテーション画像を出力するかどうか
    :param output_rec: ブロブ箇所に矩形マーキング画像を出力するかどうか
    :param rec_rotate:　ブロブ矩形を回転ありにするかどうか
    :return:
    """

    if mode == 'gray':
        gray_flg = True
    else:
        gray_flg = False

    # make binary..................
    img, img_bin = _make_binary_image(img_in,
                                      mask_low=mask_low, mask_high=mask_high,
                                      mode=mode,
                                      inverse_mask=inverse_mask,
                                      kernel_size=kernel_size,
                                      erode_iter=erode_iter,
                                      dilate_iter=dilate_iter,
                                      open_flg=open_flg,
                                      mask_img=mask_img)

    # Blob.......
    label_list, label_img = _make_blob_image(img_bin)

    group_label = None
    group_list = []
    if group_pixel > 0:
        group_bin = _make_group_image(label_list, label_img, group_pixel, group_min)
        group_list, group_label = _make_blob_image(group_bin)

    _label_num = max(len(label_list), len(group_list))
    cols, img_bin, img_seg, img_rec = _initailize_outputs(img_in, img,
                                                          _label_num,
                                                          output_seg, output_rec)

    # blob analysis..........

    res_list = []
    for label_data in label_list:
        label_no = label_data[0]
        area_label = label_data[5]

        # check area
        if (blob_min is not None) and (area_label < blob_min):
            continue
        if (blob_max is not None) and (area_label > blob_max):
            continue

        # check grouping
        if group_label is not None:
            _group_no_list = np.unique(group_label[label_img == label_no])
            if len(_group_no_list) > 0:
                group_no = _group_no_list[0]
            else:
                group_no = 0
        else:
            group_no = None

        # check blob
        defect, img_rec = _check_blob(img, label_img, label_data, img_rec, rec_rotate)
        defect.defect_no = label_no
        defect.group_no = group_no
        defect.area = area_label
        res_list.append(defect)

        # For output
        img_bin[label_img == label_no] = 255
        if output_seg:
            if group_label is not None and segment_by_group:
                _col_no = group_no
            else:
                _col_no = label_no

            if not gray_flg:
                img_seg[label_img == label_no,] = cols[_col_no]
            else:
                img_seg[label_img == label_no] = cols[_col_no][0]

        if return_if_exist:
            return img_in, res_list, img_seg, img_rec

    res_list = sorted(res_list, key=lambda x: x.area, reverse=True)

    return img_in, res_list, img_seg, img_rec


def example(debug=True):
    img_file = './image/006.png'
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)

    # threshold
    low = np.array([0, 50, 50])
    high = np.array([180, 255, 255])
    # low = 100
    # high = 200

    # mask = np.zeros_like(img)
    # mask = mask[:, :, 0]
    # mask[500:, :] = 255

    # main
    _img, res, _seg, _out = blob_analysis(img, low, high,
                                          mode='hsv',
                                          inverse_mask=False,
                                          open_flg=False,
                                          dilate_iter=1,
                                          erode_iter=1,
                                          # mask_img=mask,
                                          output_seg=True,
                                          output_rec=True,
                                          rec_rotate=False)

    cv2.imwrite('org.jpg', _img)
    cv2.imwrite('seg.jpg', _seg)
    cv2.imwrite('out.jpg', _out)

    if debug:
        for _defect in res:
            print(_defect.__dict__)


if __name__ == '__main__':
    example()
