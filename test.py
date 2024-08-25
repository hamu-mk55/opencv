import cv2
import numpy as np
from sklearn.linear_model import LinearRegression


def _make_dummy_img():
    size = 500
    sigma = 80
    x0 = 250
    y0 = 250
    intensity = 30
    offset = 50

    x = np.arange(0, size, 1)
    y = np.arange(0, size, 1)
    x, y = np.meshgrid(x, y)

    gauss = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    gauss = gauss * intensity + offset
    gauss = gauss.astype(np.uint8)

    return gauss


def _calc_shading_with_mask(img, threshold=128, mask='light'):
    # mask
    if mask == "light":
        img_mask = (img >= threshold)
    else:
        img_mask = (img <= threshold)

    img_ave = np.ma.mean(np.ma.array(img, mask=img_mask))

    # model-training
    img_h, img_w = img.shape[0:2]

    xs = np.empty((0,8))
    ys = np.empty((0))
    for x in range(0, img_w, 5):
        for y in range(0, img_h, 5):
            if img_mask[y,x]:
                continue

            xs = np.append(xs, [[x, x**2, x**3, x**4, y, y**2, y**3, y**4]], axis=0)
            ys=np.append(ys, [img[y,x]], axis=0)

    model = LinearRegression()
    model.fit(xs, ys)

    # shading-correction
    img_out = img.copy().astype(np.float32)
    for x in range(img_w):
        for y in range(img_h):
            val = img[y,x]
            pred = model.predict([[x, x**2, x**3, x**4, y, y**2, y**3, y**4]])
            ratio = img_ave / pred

            img_out[y,x] = int(val * ratio)

    img_out = np.clip(img_out, 0,255).astype(np.uint8)

    return img_out

def _equalizeHist_with_mask(img,threshold=128, mask_direction='light'):
    # mask
    if mask_direction == "light":
        img_mask = (img >= threshold)
    else:
        img_mask = (img <= threshold)

    img_1d = img[~img_mask].flatten()

    hist, bins = np.histogram(img_1d, 256, [0, 256])

    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype(np.uint8)

    img_out = cdf[img]

    return img_out

def _calc_LUT():
    x = np.arange(256)

    y = (np.sin(np.pi * (x/255 - 0.5)) + 1)/2

    y2 = y ** 2

    return y2 * 255

def main(img):
    equ = cv2.equalizeHist(img)
    cv2.imwrite("eq.jpg", equ)

    img_out = _calc_shading_with_mask(img, threshold=200)

    cv2.imwrite("org2.jpg", img_out)

    equ = cv2.equalizeHist(img_out)
    cv2.imwrite("eq2.jpg", equ)

    img_out = _equalizeHist_with_mask(img, threshold=300)
    cv2.imwrite("eq3.jpg", img_out)

    img_out = _equalizeHist_with_mask(img, threshold=200)
    cv2.imwrite("eq4.jpg", img_out)


if __name__ == '__main__':
    img = _make_dummy_img()
    cv2.circle(img, (125, 125), 40, 220, thickness=-1)
    cv2.circle(img, (375, 375), 40, 220, thickness=-1)
    cv2.circle(img, (125, 375), 40, 220, thickness=-1)


    cv2.imwrite("org.jpg", img)

    lut = _calc_LUT()

    img_lut = cv2.LUT(img, lut)
    cv2.imwrite("lut.jpg", img_lut)

    # main(img)
