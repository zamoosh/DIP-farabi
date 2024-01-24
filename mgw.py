# === mgw is stands for "motion blurring, gaussian and wiener" === #

import os
import numpy
from numpy.fft import fft2, ifft2
from scipy.signal import windows, convolve2d
import matplotlib.pyplot as plt
from library.colorful_logging import color_print

PATH = './source_images/lena-1960.jpg'
RESULT_PATH = './img_result/quran_square_result.png'


def blur(img, kernel_size: int = 3):
    dummy = numpy.copy(img)
    h = numpy.eye(kernel_size) / kernel_size
    dummy = convolve2d(dummy, h, mode='valid')
    return dummy


def add_gaussian_noise(img, sigma):
    gauss = numpy.random.normal(0, sigma, numpy.shape(img))
    noisy_img = img + gauss
    noisy_img[noisy_img < 0] = 0
    noisy_img[noisy_img > 255] = 255
    return noisy_img


def wiener_filter(img, kernel, K):
    kernel /= numpy.sum(kernel)
    dummy = numpy.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s=img.shape)
    kernel = numpy.conj(kernel) / (numpy.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = numpy.abs(ifft2(dummy))
    return dummy


def gaussian_kernel(kernel_size=3):
    h = windows.gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = numpy.dot(h, h.transpose())
    h /= numpy.sum(h)
    return h


def rgb_to_gray(rgb):
    return numpy.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def show(gray_image, blurred_img, noisy_img, filtered_img):
    # Display results
    display = [gray_image, blurred_img, noisy_img, filtered_img]
    label = [
        'Original Image',
        'Motion Blurred Image',
        'Motion Blurring + Gaussian Noise',
        'Wiener Filter applied'
    ]

    fig = plt.figure(figsize=(12, 12))

    for item in range(len(display)):
        fig.add_subplot(2, 2, item + 1)
        plt.imshow(display[item], cmap='gray')
        # plt.imshow(display[i])
        plt.title(label[item])

    plt.savefig(RESULT_PATH)
    plt.show()

    return


def main():
    file_name = os.path.join(PATH)
    gray_image = rgb_to_gray(plt.imread(file_name))

    # Blur the image
    blurred_img = blur(gray_image, kernel_size=15)

    # Add Gaussian noise
    noisy_img = add_gaussian_noise(blurred_img, sigma=20)

    # Apply Wiener Filter
    kernel = gaussian_kernel(3)
    filtered_img = wiener_filter(noisy_img, kernel, K=10)

    show(gray_image, blurred_img, noisy_img, filtered_img)

    return


if __name__ == '__main__':
    color_print(
        '*** ATTENTION: you can change the input image using the "PATH" variable at top of the file. ***'
        '\n\n',
        'red'
    )
    color_print('PROCESS IS BEING STARTED, PLEASE BE PATIENT...', 'green')
    main()
    color_print('PROCESS ENDED', 'green')
