import numpy
import cv2
import matplotlib.pyplot as plt
import scipy.io
from library.colorful_logging import color_print

PATH_1 = './source_images/blurred_low_noise.png'
PATH_2 = './source_images/blurred_mid_noise.png'
PATH_3 = './source_images/blurred_high_noise.png'
MAT_PATH = './source_images/blurred_kernel.mat'


# === constrained_ls_filter === #
def clf(image, kernel, lap, gamma: float):
    kernel /= numpy.sum(kernel)
    img_copy = numpy.copy(image)
    img_copy = numpy.fft.fftshift(numpy.fft.fft2(img_copy))
    kernel = numpy.fft.fftshift(numpy.fft.fft2(kernel, s=image.shape))
    nd_array = numpy.fft.fftshift(numpy.fft.fft2(lap, s=image.shape))
    kernel = numpy.conj(kernel) / (numpy.abs(kernel) ** 2 + gamma * nd_array)
    img_copy = img_copy * kernel
    img_copy = numpy.abs(numpy.fft.ifft2(numpy.fft.ifftshift(img_copy)))
    return img_copy


def main():
    global PATH_1, PATH_2, PATH_3, MAT_PATH

    gamma = 0.01
    number_arr = numpy.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    img_file_1 = cv2.imread(PATH_1, 0)
    img_file_2 = cv2.imread(PATH_2, 0)
    img_file_3 = cv2.imread(PATH_3, 0)

    kernel = scipy.io.loadmat(MAT_PATH).get('h')
    k = 0.01

    cls_filtering_1 = clf(img_file_1, kernel, number_arr, gamma)
    cls_filtering_2 = clf(img_file_2, kernel, number_arr, gamma)
    cls_filtering_3 = clf(img_file_3, kernel, number_arr, gamma)

    display = [img_file_1, img_file_2, img_file_3, cls_filtering_1, cls_filtering_2, cls_filtering_3]

    labels_list = [
        'Low Noise Image',
        'Med Noise Image',
        'High Noise Image',
        'cls-Low Noise o/p',
        'cls-Med Noise o/p',
        'cls-High Noise o/p'
    ]

    figure = plt.figure('Constrained least squares Filter for Image De-blurring', figsize=(15, 12))

    for item in range(len(display)):
        figure.add_subplot(2, 3, item + 1)
        plt.imshow(display[item], cmap='gray')
        plt.title(labels_list[item])

    plt.savefig('./img_result/cls_result.png')
    plt.show()

    return


if __name__ == '__main__':
    color_print('PROCESS IS BEING STARTED, PLEASE BE PATIENT...', 'green')
    main()
    color_print('PROCESS ENDED', 'green')
