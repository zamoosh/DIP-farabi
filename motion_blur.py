import cv2
import numpy
from library.colorful_logging import color_print

PATH = './source_images/quran_square.png'
RESULT_PATH = './img_result/quran_motion_blur.jpg'
IMG = cv2.imread(PATH)


def motion_blurring_3(kernel_size: int, a: float, b: float):
    global IMG

    kernel_v = numpy.zeros((kernel_size, kernel_size))
    kernel_h = numpy.copy(kernel_v)

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1) / 2)] = numpy.ones(kernel_size)
    kernel_h[int((kernel_size - 1) / 2), :] = numpy.ones(kernel_size)

    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size

    # Apply the motion blurring with parameters a and b.
    motion_kernel = a * kernel_v + b * kernel_h

    # Apply the motion blurring kernel.
    motion_blur = cv2.filter2D(IMG, -1, motion_kernel)

    # Save the output.
    cv2.imwrite(RESULT_PATH, motion_blur)

    return


def main():
    motion_blurring_3(5, 0.5, -0.1)


if __name__ == '__main__':
    color_print('PROCESS IS BEING STARTED, PLEASE BE PATIENT...', 'green')
    main()
    color_print(f'FILE SAVED IN {RESULT_PATH}', 'cyan')
