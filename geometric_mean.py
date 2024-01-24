import numpy
import cv2
from library.colorful_logging import color_print

PATH = './source_images/cameraman.png'


def show(geo_mean):
    geo_mean = numpy.uint8(geo_mean)
    cv2.imshow('1', geo_mean)
    cv2.waitKey()


def calculate_mean(img, padding, rows, cols, kernel):
    pad_img = cv2.copyMakeBorder(
        img, *[padding] * 4, cv2.BORDER_DEFAULT
    )

    geo_mean = numpy.zeros_like(img)

    for row in range(rows):
        for col in range(cols):
            geo_mean[row, col] = numpy.prod(
                pad_img[row:row + kernel, col:col + kernel]
            ) ** (1 / (kernel ** 2))

    return geo_mean


def main():
    img = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE).astype(float)

    rows, cols = img.shape[:2]
    kernel = 5
    padding = int(
        (kernel - 1) / 2
    )
    geo_mean = calculate_mean(img, padding, rows, cols, kernel)
    show(geo_mean)
    return


if __name__ == '__main__':
    color_print('PLEASE ENTER esc TO EXIT', 'bold')
    main()
    color_print('ENDED', 'green')
