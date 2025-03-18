import cnn
import scipy.io


def main():
    represent = cnn.Represent(scipy.io.loadmat("test_32x32.mat"), scipy.io.loadmat("test_32x32.mat"))
    # represent.play_data()
    represent.train_epochs()
    # represent.train_learning_rate()
    # represent.train_optimally()
    

    # represent.histogram()


if __name__ == "__main__":
    main()