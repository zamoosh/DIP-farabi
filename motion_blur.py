import cv2
import numpy as np

img = cv2.imread('./img/quran_square.png')


def motion_blurring(kernel_size: int):
    # Specify the kernel size.
    # The greater the size, the more the motion.
    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))
    kernel_h = np.copy(kernel_v)

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size

    # Apply the vertical kernel.
    vertical_mb = cv2.filter2D(img, -1, kernel_v)

    # Apply the horizontal kernel.
    horizontal_mb = cv2.filter2D(img, -1, kernel_h)

    # Save the outputs.
    cv2.imwrite('./img_result/quran_v.jpg', vertical_mb)
    cv2.imwrite('./img_result/quran_h.jpg', horizontal_mb)


def motion_blurring_2(kernel_size: int, a: float, b: float):
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel_motion_blur /= kernel_size

    # applying the kernel to the input image
    output = cv2.filter2D(img, -1, kernel_motion_blur)
    output = a * output

    cv2.imwrite('./img_result/quran_motion_blur.jpg', output)
    # cv2.imshow('Motion Blur', output)
    # cv2.waitKey(0)

    return


def motion_blurring_3(kernel_size: int, a: float, b: float):
    # Specify the kernel size.
    # The greater the size, the more the motion.
    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))
    kernel_h = np.copy(kernel_v)

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size

    # Apply the motion blurring with parameters a and b.
    motion_kernel = a * kernel_v + b * kernel_h

    # Apply the motion blurring kernel.
    motion_blur = cv2.filter2D(img, -1, motion_kernel)

    # Save the output.
    cv2.imwrite('./img_result/quran_motion_blur.jpg', motion_blur)

    return


def main():
    motion_blurring(5)
    motion_blurring_3(5, 0.5, -0.1)


if __name__ == '__main__':
    main()
