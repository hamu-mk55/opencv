import glob
import os
import shutil

import cv2
import numpy as np


def _template_match(img, template_img, mode="zncc"):
    # opencvを使用しないテンプレートマッチング
    # TODO: maskあり対応
    def _calc_correlation(_img1: np.array, _img2: np.array):
        _vec1 = np.float32(_img1.flatten())
        _vec2 = np.float32(_img2.flatten())

        if mode == 'ncc':
            # cv2.TM_CCORR_NORMEDに相当
            _upper = np.sum(_vec1 * _vec2)
            _lower = np.sqrt(np.sum(_vec1 ** 2)) * np.sqrt(np.sum(_vec2 ** 2))
            _ncc = _upper / _lower
            return _ncc

        elif mode == 'zncc':
            # cv2.TM_CCOEFF_NORMEDに相当
            _ave1 = np.mean(_vec1)
            _ave2 = np.mean(_vec2)

            _vec1 = _vec1 - _ave1
            _vec2 = _vec2 - _ave2

            _upper = np.sum(_vec1 * _vec2)
            _lower = np.sqrt(np.sum(_vec1 ** 2)) * np.sqrt(np.sum(_vec2 ** 2))
            _zncc = _upper / _lower
            return _zncc

        else:
            raise ValueError(f"illegal mode for _template-match: {mode}")

    img_h, img_w = img.shape[0:2]
    temp_h, temp_w = template_img.shape[0:2]

    score_map = np.zeros([img_h - temp_h, img_w - temp_w], dtype=np.float32)

    for y in range(img_h - temp_h):
        for x in range(img_w - temp_w):
            score_map[y, x] = _calc_correlation(img[y:y + temp_h, x:x + temp_w], template_img)

    score_map /= np.max(score_map)

    return score_map


def template_match(img: np.array,
                   temp_img: np.array,
                   is_normalized: bool = True,
                   multi_flg: bool = False,
                   multi_threshold: float = 0.5,
                   debug_file: str = None):
    """
    :param img: 対象画像
    :param temp_img: テンプレート画像
    :param is_normalized: マッチング手法で規格化するかどうか
    :param multi_flg: 複数検出するかどうか
    :param multi_threshold: 複数検出する場合の検出閾値
    :param debug_file: マッチング結果画像の出力先
    :return:
    """

    temp_h = temp_img.shape[0]
    temp_w = temp_img.shape[1]

    # テンプレートマッチング
    if is_normalized:
        method = cv2.TM_CCOEFF_NORMED
    else:
        method = cv2.TM_CCOEFF

    tp_result = cv2.matchTemplate(img, temp_img, method)
    # tp_result = _template_match(img, temp_img, mode='zncc')

    # デバッグ
    if debug_file is not None:
        cv2.imwrite(debug_file, tp_result * 255)

    # マッチング位置
    if not multi_flg:
        # マッチング最大箇所を探索
        _, max_val, _, max_loc = cv2.minMaxLoc(tp_result)

        _w0 = max_loc[0]
        _h0 = max_loc[1]
        _h_max, _w_max = tp_result.shape[0:2]

        # パラボラフィッティングでサブピクセル位置推定
        if 0 < _w0 < _w_max and 0 < _h0 < _h_max:
            _val_low = tp_result[_h0 - 1, _w0]
            _val_high = tp_result[_h0 + 1, _w0]
            _hs = (_val_low - _val_high) / (2 * _val_low - 4 * max_val + 2 * _val_high)

            _val_low = tp_result[_h0, _w0 - 1]
            _val_high = tp_result[_h0, _w0 + 1]
            _ws = (_val_low - _val_high) / (2 * _val_low - 4 * max_val + 2 * _val_high)

            top_left = (_w0 + round(_ws, 2), _h0 + round(_hs, 2))
        else:
            top_left = (_w0, _h0)

        bottom_right = (top_left[0] + int(temp_w), top_left[1] + int(temp_h))

        return max_val, top_left, bottom_right
    else:
        # マッチング率が閾値以上の点を全探索
        if not is_normalized:
            _max = np.max(tp_result)
            multi_threshold = _max * multi_threshold

        loc = np.where(tp_result >= multi_threshold)

        results = []
        for pt in zip(loc[1], loc[0]):
            _w0 = pt[0]
            _h0 = pt[1]
            _h_max, _w_max = tp_result.shape[0:2]

            top_left = (_w0, _h0)
            bottom_right = (top_left[0] + int(temp_w), top_left[1] + int(temp_h))
            val = tp_result[_h0, _w0]

            # 5x5近傍でマッチング率が最大の場合に、探索対象と判定
            max_val_in_nears = np.max(
                tp_result[
                max(_h0 - 2, 0):min(_h0 + 2, _h_max),
                max(_w0 - 2, 0):min(_w0 + 2, _w_max)]
            )

            if val == max_val_in_nears:
                results.append([val, top_left, bottom_right])

        return results


class TemplateMatch:
    def __init__(self, resize_ratio: float = 1.0, gray_flg: bool = False):
        self.resize_ratio: float = resize_ratio
        self.gray_flg: bool = gray_flg

        self.temp_img: np.array = None
        self.img: np.array = None
        self.results: list = []

        self._init_params()

    def load_template(self, temp_path: str):
        _img = self._read_img(temp_path)
        _img = self._img_cvt(_img)
        self.temp_img = _img

        return True

    def exec(self, img_path=None, img=None,
             is_normalized=False,
             multi_flg=False,
             multi_threshold=0.9,
             rectangle_flg=False):
        self._init_params()

        # 画像ファイルの読み込み
        if img is None:
            if img_path is not None:
                self.img = self._read_img(img_path=img_path)
            else:
                raise ValueError("No input for exec")
        else:
            self.img = img

        img = self._img_cvt(self.img)

        # テンプレートマッチング
        if not multi_flg:
            _val, _top_left, _bottom_right = template_match(img, self.temp_img,
                                                            is_normalized=is_normalized)

            _top_left = (_top_left[0] / self.resize_ratio, _top_left[1] / self.resize_ratio)
            _bottom_right = (_bottom_right[0] / self.resize_ratio, _bottom_right[1] / self.resize_ratio)

            self.results = [_val, _top_left, _bottom_right]

            if rectangle_flg:
                _top_left = (int(_top_left[0]), int(_top_left[1]))
                _bottom_right = (int(_bottom_right[0]), int(_bottom_right[1]))

                cv2.rectangle(self.img, _top_left, _bottom_right, (255, 255, 255), thickness=5)
                cv2.rectangle(self.img, _top_left, _bottom_right, (255, 0, 0), thickness=3)

            return self.img

        else:
            _results = template_match(img, self.temp_img,
                                      is_normalized=is_normalized,
                                      multi_flg=multi_flg,
                                      multi_threshold=multi_threshold)

            self.results = []
            for res in _results:
                _val = res[0]
                _top_left = res[1]
                _bottom_right = res[2]

                _top_left = (_top_left[0] / self.resize_ratio, _top_left[1] / self.resize_ratio)
                _bottom_right = (_bottom_right[0] / self.resize_ratio, _bottom_right[1] / self.resize_ratio)

                self.results.append([_val, _top_left, _bottom_right])

                if rectangle_flg:
                    _top_left = (int(_top_left[0]), int(_top_left[1]))
                    _bottom_right = (int(_bottom_right[0]), int(_bottom_right[1]))

                    cv2.rectangle(self.img, _top_left, _bottom_right, (255, 255, 255), thickness=5)
                    cv2.rectangle(self.img, _top_left, _bottom_right, (255, 0, 0), thickness=3)

            return self.img

    def _read_img(self, img_path):
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f'No Image file: {img_path}')

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        return img

    def _init_params(self):
        self.img = None
        self.results = None

    def _img_cvt(self, img):
        if img is None:
            raise ValueError("img is None")

        _h = img.shape[0]
        _w = img.shape[1]

        if self.gray_flg:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img,
                         (int(_w * self.resize_ratio), int(_h * self.resize_ratio)),
                         cv2.INTER_LINEAR)

        return img


if __name__ == '__main__':
    app = TemplateMatch(gray_flg=True, resize_ratio=1.0)
    app.load_template("template.jpg")

    for img_path in glob.glob('sample.jpg'):
        out = app.exec(img_path=img_path,
                       is_normalized=True,
                       multi_flg=False,
                       multi_threshold=0.8,
                       rectangle_flg=True)
        print(img_path, app.results)

        cv2.imwrite("out.jpg", out)
