import project
import time

def main():
    nn_model = project.train_nn()
    nn_model.basic_train()
    # nn_model.repesent_learning_rate()
    # nn_model.represent_layers()
    # nn_model.represent_optimal()
    
if __name__ == "__main__":
    main()