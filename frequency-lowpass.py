import cv2
import numpy
from library.colorful_logging import color_print

PATH = './source_images/characters.tif'


def lowpass(img, size: int, do) -> tuple[numpy.ndarray, numpy.ndarray]:
    d_var = numpy.zeros([size, size], dtype=numpy.uint32)
    h_var = numpy.zeros([size, size], dtype=numpy.uint8)

    r_var = img.shape[0] // 2
    c_var = img.shape[1] // 2

    for u in range(0, size):
        for v in range(0, size):
            d_var[u, v] = numpy.sqrt(
                (u - r_var) ** 2 + (v - c_var) ** 2
            )

    for item in range(size):
        for sec_item in range(size):
            if d_var[item, sec_item] > do:
                h_var[item, sec_item] = 0
            else:
                h_var[item, sec_item] = 255

    return h_var, d_var


def main():
    global PATH

    image = cv2.imread(PATH, 0)
    size = image.shape[0]

    first_img, second_img = lowpass(image, size, 80)

    inp = numpy.fft.fftshift(
        numpy.fft.fft2(image)
    )
    out = inp * first_img
    out = numpy.abs(numpy.fft.ifft2(numpy.fft.ifftshift(out)))
    out = numpy.uint8(cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX, -1))

    cv2.imshow('Filter Representation', first_img)
    cv2.waitKey(0)

    cv2.imshow('Magnitude Spectrum', cv2.normalize(numpy.abs(inp), None, 0, 255, cv2.NORM_MINMAX, -1))
    cv2.waitKey(0)

    cv2.imshow('Ideal Low Pass Filtered Output', out)
    cv2.waitKey(0)

    return


if __name__ == '__main__':
    color_print('PLEASE ENTER esc TO EXIT', 'bold')
    main()
    color_print('ENDED', 'green')
