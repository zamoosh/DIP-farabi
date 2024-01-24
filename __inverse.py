import numpy
import cv2
import matplotlib.pyplot as plt
import scipy.io

path4 = './source_images/blurred_low_noise.png'
path5 = './source_images/blurred_mid_noise.png'
path6 = './source_images/blurred_high_noise.png'


def inverse_filter(img, kernel, k):
    kernel /= numpy.sum(kernel)
    dummy = numpy.copy(img)
    dummy = numpy.fft.fftshift(numpy.fft.fft2(dummy))
    kernel = numpy.fft.fftshift(numpy.fft.fft2(kernel, s=img.shape))
    kernel = numpy.conj(kernel) / (numpy.abs(kernel) ** 2 + k)
    dummy = dummy * kernel
    dummy = numpy.abs(numpy.fft.ifft2(numpy.fft.ifftshift(dummy)))
    return dummy


[image1, image2, image3] = cv2.imread(path4, 0), cv2.imread(path5, 0), cv2.imread(path6, 0)
mat = scipy.io.loadmat('./source_images/blurred_kernel.mat')
kernel = mat.get('h')
K = 0.01
[inv1, inv2, inv3] = inverse_filter(image1, kernel, K), inverse_filter(image2, kernel, K), inverse_filter(image3,
                                                                                                          kernel, K)
# Image was deblurred from the input. For the sake of completeness need a smoothing filter to ensure the image obtained is more clean
display11 = [image1, image2, image3, inv1, inv2, inv3]
label11 = ['Low Noise Image', 'Med Noise Image', 'High Noise Image', 'inv-Low Noise o/p', 'inv-Med Noise o/p',
           'inv-High Noise o/p']
fig11 = plt.figure('Inverse Filter for Image Deblurring', figsize=(12, 10))
for i in range(len(display11)):
    fig11.add_subplot(2, 3, i + 1)
    plt.imshow(display11[i], cmap='gray')
    plt.title(label11[i])
plt.show()


def main():
    pass


if __name__ == '__main__':
    main()
