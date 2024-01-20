import random

import cv2
import numpy as np


def blob_analysis(img_bgr,
                  mask_low=None, mask_high=None,
                  mode='hsv', inverse_mask=False,
                  kernel_size=3, erode_iter=0, dilate_iter=0,
                  open_flg=True,
                  blob_min=10, blob_max=None,
                  mask_img=None,
                  return_if_exist=False,
                  output_seg=False,
                  output_rec=False, rec_rotate=False):
    """
    ブロブ解析を実行
    :param img_bgr: 入力画像データ(BGR形式 or モノクロ画像)
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

    # Check Params.......
    # caution: Hue-range is [0:180] in open-cv, [0:255] in image-J
    if mask_low is None:
        raise ValueError('blob_analysis: mask-low is None')
    if mask_high is None:
        raise ValueError('blob_analysis: mask-high is None')

    gray_flg = False
    if mode == 'hsv':
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    elif mode == 'bgr':
        img = img_bgr.copy()
    elif mode == 'gray':
        gray_flg = True
        if img_bgr.ndim == 3:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img = img_bgr.copy()
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

    # Blob Analysis.......
    label = cv2.connectedComponentsWithStats(img_bin)

    n_label = label[0] - 1
    label_img = label[1]
    # notes: data_label=[cnt, (x0,y0,width,height,area)]
    data_label = np.delete(label[2], 0, 0)

    # For output
    if output_seg:
        cols = []
        for cnt in range(n_label):
            cols.append(np.array([random.randint(100, 255),
                                  random.randint(100, 255),
                                  random.randint(100, 255)]))
        img_seg = np.zeros(img.shape)
    else:
        img_seg = None

    if output_rec:
        img_out = img_bgr.copy()
    else:
        img_out = None

    # Check blob..........
    res_list = []
    for cnt in range(n_label):
        area_label = data_label[cnt, 4]

        # check area
        if (blob_min is not None) and (area_label < blob_min):
            continue
        if (blob_max is not None) and (area_label > blob_max):
            continue

        # notes: cv2.inRange is faster than np.where
        _img = cv2.inRange(label_img, cnt + 1, cnt + 1)

        # notes: ret is 2 in opencv4 (3 in opencv3)
        contours, _ = cv2.findContours(_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # search outside-contours(=contour area is maximum)
        _max_contour = None
        _max_area = 0
        for cont_cnt, contour in enumerate(contours):
            area_contour = cv2.contourArea(contour)

            if area_contour > _max_area:
                _max_area = area_contour
                _max_contour = contour

        # Min-Rectangle
        if not rec_rotate:
            x, y, w, h = cv2.boundingRect(_max_contour)
            res_list.append([area_label, _max_area, x, y, w, h])

            if output_rec:
                cv2.rectangle(img_out, (x, y), (x + w, y + h),
                              color=0, thickness=3)
                cv2.rectangle(img_out, (x, y), (x + w, y + h),
                              color=255, thickness=2)
        else:
            # notes: rect = [(x, y), (w, h), angle]
            _rect = cv2.minAreaRect(_max_contour)
            _box = cv2.boxPoints(_rect)
            _box = np.int0(_box)

            def _f(input_num):
                return round(input_num, 1)

            x = _rect[0][0]
            y = _rect[0][1]
            h = max(_rect[1][0], _rect[1][1])
            w = min(_rect[1][0], _rect[1][1])
            res_list.append([area_label, _max_area, _f(x), _f(y), _f(w), _f(h)])

            if output_rec:
                img_out = cv2.drawContours(img_out, [_box], 0,
                                           color=0, thickness=3)
                img_out = cv2.drawContours(img_out, [_box], 0,
                                           color=255, thickness=2)

        # For output
        if output_seg:
            if not gray_flg:
                img_seg[label_img == cnt + 1,] = cols[cnt]
            else:
                img_seg[label_img == cnt + 1] = cols[cnt][0]

        if return_if_exist:
            return img_bgr, res_list, img_seg, img_out

    res_list = sorted(res_list, key=lambda x: x[0], reverse=True)

    return img_bgr, res_list, img_seg, img_out


def example(debug=True):
    img_file = './image/NG/000.png'
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    print(img.shape)

    # threshold
    low = np.array([0, 50, 0])
    high = np.array([180, 150, 255])
    # low = 100
    # high = 200

    mask = np.zeros_like(img)
    mask = mask[:,:,0]
    mask[500:, :] = 255

    # main
    _img, res, _seg, _out = blob_analysis(img, low, high,
                                          mode='hsv',
                                          inverse_mask=True,
                                          dilate_iter=1,
                                          mask_img=mask,
                                          output_seg=True,
                                          output_rec=True,
                                          rec_rotate=False)

    cv2.imwrite('org.jpg', _img)
    cv2.imwrite('seg.jpg', _seg)
    cv2.imwrite('out.jpg', _out)

    if debug:
        for _res in res:
            print(_res)


if __name__ == '__main__':
    example()
