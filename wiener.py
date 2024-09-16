import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.signal import gaussian
from skimage import restoration



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

def main(img):
    # 各パラメータ
    psf_px = 51  # PSF用のガウシアンフィルタのピクセルサイズ(奇数のみ)
    psf_sigma = 6  # PSF用のガウシアンフィルタのσ
    noise_sigma = 0.5  # ガウシアンノイズのσ
    iterations = 200  # RL法の反復回数
    balance = 2.2

    # PSF
    psf = get_psf(psf_px, psf_sigma)

    # regに4近傍ラプラシアンフィルタを使用(デフォルトのreg=Noneと同一のフィルタ)
    laplacian4_reg = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    img_wiener4 = restoration.wiener(img / 255., psf, balance, laplacian4_reg)
    img_wiener4 *= 255.
    img_wiener4 = img_wiener4.astype(np.uint8)

    # regに8近傍ラプラシアンフィルタを使用
    laplacian8_reg = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)
    img_wiener8 = restoration.wiener(img / 255., psf, balance, laplacian8_reg)
    img_wiener8 *= 255.
    img_wiener8 = img_wiener8.astype(np.uint8)

    # 比較用にRichardson-Lucy deconvolution
    img_rl = restoration.richardson_lucy(img / 255., psf, iterations)
    img_rl *= 255.
    img_rl = img_rl.astype(np.uint8)

    cv2.imwrite("img_w4.jpg", img_wiener4)
    cv2.imwrite("img_w8.jpg", img_wiener8)
    cv2.imwrite("img_rl.jpg", img_rl)

if __name__ == '__main__':
    img = _make_dummy_img()
    cv2.circle(img, (250, 250), 30, 220, thickness=-1)

    # 学習
    img11 = cv2.GaussianBlur(img, ksize=(21, 21), sigmaX=0)

    noise = np.random.randint(0, 1, (500, 500))
    img11 = img11 + noise

    cv2.imwrite("org.jpg", img11)

    psf_px = 51  # PSF用のガウシアンフィルタのピクセルサイズ(奇数のみ)
    psf_sigma = 5  # PSF用のガウシアンフィルタのσ
    psf = get_psf(psf_px, psf_sigma)
    img11 = cv2.filter2D(img11, -1, psf)
    cv2.imwrite("img_blur.jpg", img11)

    main(img11)

