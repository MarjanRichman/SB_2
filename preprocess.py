import cv2
import glob
import numpy as np
import os


def histogram_eq(img):
    changed_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    changed_img[:, :, 0] = cv2.equalizeHist(changed_img[:, :, 0])
    changed_img = cv2.cvtColor(changed_img, cv2.COLOR_YCrCb2BGR)
    return changed_img


def sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp_img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return sharp_img


def gamma_correction(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


if __name__ == '__main__':

    images_path = glob.glob(r"C:/Users/Marjan/Desktop/SB 2. assignment/data/ears/test/*.png")
    for img in images_path:
        orig_name = os.path.basename(img)
        print(orig_name)
        orig = cv2.imread(img)
        changed_image = gamma_correction(orig, 2.2)
        changed_image = sharpen(changed_image)
        changed_image = histogram_eq(changed_image)
        # cv2.imshow("Original", orig)
        # cv2.imshow("Changed", changed_image)
        # key = cv2.waitKey(0)
        cv2.imwrite(orig_name, changed_image)
