import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./img/cameraman.png', cv2.IMREAD_GRAYSCALE).astype(float)
rows, cols = img.shape[:2]
ksize = 5

padsize = int((ksize - 1) / 2)
pad_img = cv2.copyMakeBorder(img, *[padsize] * 4, cv2.BORDER_DEFAULT)
geomean1 = np.zeros_like(img)
for r in range(rows):
    for c in range(cols):
        geomean1[r, c] = np.prod(pad_img[r:r + ksize, c:c + ksize]) ** (1 / (ksize ** 2))
geomean1 = np.uint8(geomean1)
cv2.imshow('1', geomean1)
cv2.waitKey()

# geomean2 = np.uint8(np.exp(cv2.boxFilter(np.log(img), -1, (ksize, ksize))))
# plt.imshow(geomean2, cmap='gray')
# cv2.imshow('2', geomean2)
# cv2.waitKey()

