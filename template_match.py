import glob
import os
import shutil

import cv2
import numpy as np


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

    # デバッグ
    if debug_file is not None:
        cv2.imwrite(debug_file, tp_result * 255)

    # マッチング位置
    if not multi_flg:
        _, max_val, _, max_loc = cv2.minMaxLoc(tp_result)

        top_left = (max_loc[0], max_loc[1])
        bottom_right = (top_left[0] + int(temp_w), top_left[1] + int(temp_h))

        return max_val, top_left, bottom_right
    else:
        if not is_normalized:
            _max = np.max(tp_result)
            multi_threshold = _max * multi_threshold

        loc = np.where(tp_result >= multi_threshold)

        results = []
        for pt in zip(loc[1], loc[0]):
            top_left = (pt[0], pt[1])
            bottom_right = (top_left[0] + int(temp_w), top_left[1] + int(temp_h))

            results.append([None, top_left, bottom_right])

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

            _top_left = (int(_top_left[0] / self.resize_ratio), int(_top_left[1] / self.resize_ratio))
            _bottom_right = (int(_bottom_right[0] / self.resize_ratio), int(_bottom_right[1] / self.resize_ratio))

            self.results = [_val, _top_left, _bottom_right]

            if rectangle_flg:
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
                _top_left = res[1]
                _bottom_right = res[2]

                _top_left = (int(_top_left[0] / self.resize_ratio), int(_top_left[1] / self.resize_ratio))
                _bottom_right = (int(_bottom_right[0] / self.resize_ratio), int(_bottom_right[1] / self.resize_ratio))

                self.results.append([None, _top_left, _bottom_right])

                if rectangle_flg:
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
                       multi_flg=True,
                       multi_threshold=0.5,
                       rectangle_flg=True)
        print(img_path, app.results)

        cv2.imwrite("out.jpg", out)
