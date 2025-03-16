import cnn
import scipy.io


def main():
    represent = cnn.Represent(scipy.io.loadmat("test_32x32.mat"), scipy.io.loadmat("test_32x32.mat"))
    represent.train()
    represent.shape_of()
    # represent.histogram()


if __name__ == "__main__":
    main()