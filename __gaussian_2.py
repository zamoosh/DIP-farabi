import numpy as np
import cv2

img_path = './source_images/quran_square.png'

img = cv2.imread(img_path)
mean = 0
var = 1000
sigma = var ** 0.5
# gaussian = np.random.normal(mean, sigma, (224, 224))  # np.zeros((224, 224), np.float32)
gaussian = np.random.normal(mean, sigma, (5, 5))

noisy_image = np.zeros(img.shape, np.float32)

if len(img.shape) == 2:
    noisy_image = img + gaussian
else:
    noisy_image[:, :, 0] = img[:, :, 0] + gaussian
    noisy_image[:, :, 1] = img[:, :, 1] + gaussian
    noisy_image[:, :, 2] = img[:, :, 2] + gaussian

cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
noisy_image = noisy_image.astype(np.uint8)

cv2.imshow("./source_images", img)
cv2.imshow("gaussian", gaussian)
cv2.imshow("noisy", noisy_image)

cv2.waitKey(0)
