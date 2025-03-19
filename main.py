import project


def main():
    nn_model = project.train_nn()

    nn_model.repesent_learning_rate()
    nn_model.represent_layers()

if __name__ == "__main__":
    main()