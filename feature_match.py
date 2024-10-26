import glob
import os
import shutil

import cv2
import numpy as np


def feature_match(img, template_img, mode="ORB",
                  match_num=None, debug=True):
    if mode == "AKAZE":
        detector = cv2.AKAZE_create()
    else:
        detector = cv2.ORB_create()

    # detect features
    kp_template, des_template = detector.detectAndCompute(template_img, None)
    kp, des = detector.detectAndCompute(img, None)

    # matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des_template, des)
    matches = sorted(matches, key=lambda x: x.distance)

    if debug:
        for cnt, match in enumerate(matches):
            _kp = kp[match.queryIdx]
            print(match.distance, _kp.pt)

            if cnt > 10:
                break

    # output
    if len(matches) < 4:
        raise ValueError(f"match-points is too low: {len(matches)}")
    elif len(matches) < match_num:
        match_num = len(matches)

    if match_num is None:
        match_selected = matches
    else:
        match_selected = matches[:match_num]

    img_matched = cv2.drawMatches(template_img, kp_template, img, kp,
                                  match_selected, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    template_pts = np.float32([kp_template[m.queryIdx].pt for m in match_selected]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp[m.trainIdx].pt for m in match_selected]).reshape(-1, 1, 2)

    # template -> target
    Matrix_t2d, _ = cv2.findHomography(template_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = template_img.shape[0:2]

    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    detect_points = cv2.perspectiveTransform(pts, Matrix_t2d)

    # target -> template
    Matrix_d2t, _ = cv2.findHomography(dst_pts, template_pts, cv2.RANSAC, 5.0)
    img_aligned = cv2.warpPerspective(img, Matrix_d2t, (w, h))

    return detect_points, img_aligned, img_matched


class FeatureMatch:
    def __init__(self, gray_flg: bool = False):
        self.gray_flg = gray_flg

        self.temp_img: np.array = None
        self.img: np.array = None
        self.results: list = []

        self.img_aligned: np.array = None
        self.img_matched: np.array = None

        self._init_params()

    def load_template(self, temp_path: str):
        _img = self._read_img(temp_path)
        _img = self._img_cvt(_img)
        self.temp_img = _img

        return True

    def exec(self,
             img_path=None,
             img=None,
             mode="ORB",
             match_num=5,
             rectangle_flg=True,
             debug=False):

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

        # 特徴量マッチング
        detect_points, img_aligned, img_matched = feature_match(img, self.temp_img,
                                                                mode=mode,
                                                                match_num=match_num,
                                                                debug=debug)

        self.results = detect_points
        self.img_aligned = img_aligned
        self.img_matched = img_matched

        if rectangle_flg:
            img = cv2.polylines(img, [np.int32(detect_points)], True, 255, 3, cv2.LINE_AA)

        return img

    def _read_img(self, img_path):
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f'No Image file: {img_path}')

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        return img

    def _init_params(self):
        self.img = None
        self.results = None
        self.img_aligned: np.array = None
        self.img_matched: np.array = None

    def _img_cvt(self, img):
        if img is None:
            raise ValueError("img is None")

        _h = img.shape[0]
        _w = img.shape[1]

        if self.gray_flg:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img


if __name__ == '__main__':
    app = FeatureMatch(gray_flg=True)
    app.load_template("box.png")

    for img_path in glob.glob('box_in_scene.png'):
        out = app.exec(img_path=img_path,
                       mode="AKAZE",
                       match_num=20,
                       rectangle_flg=True,
                       debug=False)
        print(img_path)

        cv2.imwrite("out.jpg", out)
        cv2.imwrite("out2.jpg", app.img_aligned)
        cv2.imwrite("out3.jpg", app.img_matched)
