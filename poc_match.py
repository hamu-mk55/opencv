import glob
import os
import shutil

import cv2
import numpy as np


def poc_match(img: np.array,
                   temp_img: np.array):
    """
    :param img: 対象画像
    :param temp_img: テンプレート画像
    :return:
    """

    (x, y), response = cv2.phaseCorrelate(img.astype(np.float32), temp_img.astype(np.float32))

    return x,y


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

    def exec(self, img_path=None, img=None):
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
        x,y=poc_match(img, self.temp_img)

        return x,y

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
    app.load_template("org.jpg")

    for img_path in glob.glob('./images3/*.jpg'):
        x,y=app.exec(img_path=img_path)

        print(os.path.basename(img_path), x, y)