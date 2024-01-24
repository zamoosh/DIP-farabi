import skimage
import matplotlib.pyplot as plt


class Gaussian:
    ROWS = 4
    COLS = 2
    GAUSSIAN = 1
    LOCALVAR = 2
    POISSON = 3
    SALT = 4
    PEPPER = 5
    SANDP = 6
    SPECKLE = 7
    ORIGINAL = 8
    GAUSSIAN_MODE = 'gaussian'
    LOCALVAR_MODE = 'localvar'
    POISSON_MODE = 'poisson'
    SALT_MODE = 'salt'
    PEPPER_MODE = 'pepper'
    SALT_PEPPER_MODE = 's&p'
    SPECKLE_MODE = 'speckle'

    def __init__(self, img_path: str, figure_size: tuple[int, int] = (30, 30)):
        plt.figure(figsize=figure_size)
        self.img_path = img_path
        self.img = None

    def save(self):
        plt.savefig("./img_result/result.png")
        plt.close()

    def plot_noise(self, mode: str | None, row, col, i):
        plt.subplot(row, col, i)
        if mode is not None:
            gimg = skimage.util.random_noise(self.img, mode=mode)
            plt.imshow(gimg)
        else:
            plt.imshow(self.img)
        plt.title(mode)
        plt.axis("off")

    def process(self):
        self.img = skimage.io.imread(self.img_path) / 255.0
        self.prepare_plot()

    def prepare_plot(self):
        self.plot_noise(self.GAUSSIAN_MODE, 1, 1, self.GAUSSIAN)
        # self.plot_noise(self.GAUSSIAN_MODE, self.ROWS, self.COLS, self.GAUSSIAN)
        # self.plot_noise(self.LOCALVAR_MODE, self.ROWS, self.COLS, self.LOCALVAR)
        # self.plot_noise(self.POISSON_MODE, self.ROWS, self.COLS, self.POISSON)
        # self.plot_noise(self.SALT_MODE, self.ROWS, self.COLS, self.SALT)
        # self.plot_noise(self.PEPPER_MODE, self.ROWS, self.COLS, self.PEPPER)
        # self.plot_noise(self.SALT_PEPPER_MODE, self.ROWS, self.COLS, self.SANDP)
        # self.plot_noise(self.SPECKLE_MODE, self.ROWS, self.COLS, self.SPECKLE)
        # self.plot_noise(None, self.ROWS, self.COLS, self.ORIGINAL)
        self.save()
        # plt.show()


def plot_noise(img, mode, r, c, i):
    plt.subplot(r, c, i)
    if mode is not None:
        gimg = skimage.util.random_noise(img, mode=mode)
        plt.imshow(gimg)
    else:
        plt.imshow(img)
    plt.title(mode)
    plt.axis("off")
    plt.savefig("./img_result/result.png")


def main():
    # img_path = "./source_images/quran.png"
    img_path = "./img_result/quran_motion_blur.jpg"
    g = Gaussian(img_path, (40, 40))
    g.process()


if __name__ == '__main__':
    main()
