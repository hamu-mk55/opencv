import os

import cv2
import numpy as np
from scipy.stats import multivariate_normal

def _make_dummy_img(sigma=80, intensity=0, offset=50, img_astype=np.uint8):
    size = 500
    x0 = 250
    y0 = 250

    x = np.arange(0, size, 1)
    y = np.arange(0, size, 1)
    x, y = np.meshgrid(x, y)

    gauss = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    gauss = gauss * intensity + offset
    gauss = gauss.astype(img_astype)

    return gauss

def get_psf(px, sigma):
    half_px = px // 2
    x, y = np.mgrid[-half_px:half_px+1:, -half_px:half_px+1:]
    pos = np.dstack((x, y))
    mean = np.array([0, 0])  # 分布の平均
    cov = np.array([[sigma**2, 0], [0, sigma**2]])  # 分布の分散共分散行列
    rv = multivariate_normal(mean, cov)
    psf = rv.pdf(pos)
    return psf

def fft_test(img):
    fft_img = np.fft.fft2(img)
    fft_img = np.fft.fftshift(fft_img)

    max_real = np.max(fft_img.real)
    max_imag = np.max(fft_img.imag)
    max_abs = np.max(np.abs(fft_img))

    fft_abs = (np.abs(fft_img) / max_abs * 255 * 10).astype(np.uint8)
    fft_real = (fft_img.real / max_real * 255 * 10).astype(np.uint8)
    fft_imag = (fft_img.imag / max_imag * 255 * 10).astype(np.uint8)

    cv2.imwrite("fft_abs.jpg", fft_abs)
    cv2.imwrite("fft_real.jpg", fft_real)
    cv2.imwrite("fft_imag.jpg", fft_imag)

    mask = _make_dummy_img(sigma=100, intensity=1, offset=0, img_astype=np.float32)

    ifft_img = np.fft.ifftshift(fft_img * mask)
    ifft_img = np.fft.ifft2(ifft_img)
    ifft_img = np.abs(ifft_img).astype(np.uint8)

    cv2.imwrite("ifft.jpg", ifft_img)


class ImageResults:
    def __init__(self, cnt, ksize, max2min, offset):
        self.ksize = ksize
        self.max2min = max2min
        self.offset = offset
        self.cnt = cnt


def diff_split(img, out_dir = None):
    def _split_imgs_by_freq(img_org, ksize, img_out_offset=None):
        img_base = cv2.GaussianBlur(img_org, ksize=(ksize, ksize), sigmaX=0)
        img_diff = img_org - img_base

        if img_out_offset is not None:
            img_base_out = img_base.copy()
            img_base_out = np.clip((img_base_out + img_out_offset), 0, 255).astype(np.uint8)
        else:
            img_base_out = None

        _max = np.max(img_base)
        _min = np.min(img_base)
        max2min = _max - _min

        return img_base, img_diff, img_base_out, max2min

    img = img.astype(np.float32)

    results_list = []

    # 1st
    img_base, img_diff, img_out, max2min = _split_imgs_by_freq(img, ksize=31, img_out_offset=0)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(f"{out_dir}/img0.jpg", img_out)
        # print(f"{max2min}")

    results_list.append(ImageResults(cnt=0, ksize=31, max2min=max2min, offset=0))

    # 2nd
    for cnt, ksize in enumerate(range(29, 0, -2)):
        img_base, img_diff, img_out, max2min = _split_imgs_by_freq(img_diff, ksize=ksize, img_out_offset=128)

        if out_dir is not None:
            cv2.imwrite(f"{out_dir}/img{cnt+1:02d}.jpg", img_out)
            # print(f"{max2min}")

        results_list.append(ImageResults(cnt = cnt+1, ksize=ksize, max2min=max2min, offset=128))

    return results_list


def diff_merge(img, res_list):
    def _merge_imgs_by_freq(img_org, img_out, res: ImageResults):

        img_base = cv2.GaussianBlur(img_org, ksize=(res.ksize, res.ksize), sigmaX=0)
        img_diff = img_org - img_base

        _max = np.max(img_base)
        _min = np.min(img_base)
        max2min = _max - _min

        ratio = res.max2min / max2min

        img_base = img_base * ratio

        print(f'\t, {res.max2min / max2min}')

        if img_out is None:
            img_out = img_base
        else:
            img_out = img_out + img_base

        return img_diff, img_out

    img = img.astype(np.float32)

    img_out = None
    for cnt, res in enumerate(res_list):
        print(cnt, res.ksize)

        if cnt == 0:
            img_diff, img_out = _merge_imgs_by_freq(img, img_out, res)

        else:
            img_diff, img_out = _merge_imgs_by_freq(img_diff, img_out, res)

        cv2.imwrite(f"img{cnt:02d}.jpg", np.clip(img_out, 0, 255).astype(np.uint8))


if __name__ == '__main__':
    img = _make_dummy_img()
    cv2.circle(img, (250, 250), 30, 220, thickness=-1)

    # 学習
    img11 = cv2.GaussianBlur(img, ksize=(21, 21), sigmaX=0)

    noise = np.random.randint(0, 10, (500, 500))
    img11 = img11 + noise

    cv2.imwrite("org.jpg", img11)

    res_list = diff_split(img11, out_dir='./debug')

    # 復元処理
    img22 = cv2.GaussianBlur(img, ksize=(41, 41), sigmaX=0)

    noise = np.random.randint(0, 1, (500, 500))
    img22 = img22 + noise

    cv2.imwrite("org22.jpg", img22)
    diff_merge(img22, res_list)
