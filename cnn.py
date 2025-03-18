import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import scipy.io
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


class CNN(object):
    def __init__(self):
        # Updated for 3-channel 32x32 images
        self.conv_ = torch.nn.Conv2d(3, 16, (5,5), stride=2)  
        self.pool_ = torch.nn.MaxPool2d(3, stride=2)          
        self.lin_ = torch.nn.Linear(576, 10)  # Adjusted for new feature size
            
    def forward_(self, X):
        """Compute NN forward pass and output class probabilities"""
        r1 = self.conv_(X)          
        h1 = torch.relu(r1)         
        h1_pooled = self.pool_(h1)  
        h1_flat = torch.nn.Flatten()(h1_pooled)  
        r2 = self.lin_(h1_flat)
        f  = torch.softmax(r2, dim=1)  
        return f 
        
    def parameters(self): 
        return list(self.conv_.parameters()) + list(self.lin_.parameters())
    
    def predict(self, X):
        """Compute NN class predictions"""
        Xtorch = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
        return np.argmax(self.forward_(Xtorch).detach().numpy(), axis=1)

    def J01(self, X, y):  
        return (y != self.predict(X)).mean()

    def JNLL_(self, X, y):  
        return (y != self.predict(X)).mean()

    def fit(self, X, y, batch_size=256, max_iter=100, learning_rate_init=0.005, plot=False):
        self.classes_ = np.unique(y)        

        Xtorch = torch.tensor(X).permute(0, 3, 1, 2).float()  # Reshape correctly
        ytorch = torch.tensor(y, dtype=torch.long)

        self.loss01, self.lossNLL = [self.J01(X, y)], [float(self.JNLL_(Xtorch, ytorch))]

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate_init)

        m = X.shape[0]
        for epoch in range(max_iter):                        
            pi = np.random.permutation(m)                   
            for i in range(0, m, batch_size):    
                ivals = pi[i:i+batch_size]
                optimizer.zero_grad()                        
                Ji = self.JNLL_(Xtorch[ivals], ytorch[ivals])
                Ji.backward()
                optimizer.step()
            self.loss01.append(self.J01(X, y))              
            self.lossNLL.append(float(self.JNLL_(Xtorch, ytorch)))

            if plot:                                         
                plt.plot(range(epoch+2), self.loss01, 'b-', range(epoch+2), self.lossNLL, 'c-')
                plt.title(f'J01: {self.loss01[-1]}, NLL: {self.lossNLL[-1]}')
                plt.draw(); plt.pause(.01)


class Represent:
    def __init__(self, training, validation):
        self.training_data_X = training["X"].transpose(3, 0, 1, 2)
        self.training_data_y = training["y"]
        self.validation_data_X = validation["X"].transpose(3, 0, 1, 2)
        self.validation_data_y = validation["y"]

    def shape_of(self):
        # CONTEXT: 
        # (32, 32) is our image array
        # (3) is our channel, when you access one of our pixels, it'll be a (3, 1) array, each index as R-G-B
        # (26032) is our amount of images
        print(f"Training Data X Shape: {self.training_data_X.shape}")
        print(f"Validation Data X Shape: {self.validation_data_X.shape}")
        
        # CONTEXT:
        # 26032 classifications for 26032 images
        print(f"Training Data y Shape: {self.training_data_y.shape}")
        print(f"Validation Data y Shape: {self.validation_data_y.shape}")

        print(f"\nClasses: {np.unique(self.validation_data_y)}")

    def play_data(self):
        print(self.training_data_X.shape)
        print(self.training_data_y[0])

    # Show our data
    def histogram_training_labels(self):
        plt.hist(self.training_data_y, bins=np.arange(11))
        plt.xlabel("House Number")
        plt.ylabel("Count")
        plt.title("Training data labels")
        plt.show()


    def histogram_validation_labels(self):
        plt.hist(self.validation_data_y, bins=np.arange(1, 11))
        plt.xlabel("House Number")
        plt.ylabel("Count")
        plt.title("Training data labels")
        plt.show()


    def train_epochs(self):
        # epochs = [1, 10, 50, 75, 100, 125]
        epochs = [25, 50]

        tr_err_rate = []
        ve_err_rate = []

        for epoch in epochs:
            print(epoch)
            cnn = CNN()
            cnn.fit(self.training_data_X[:1000], self.training_data_y[:1000], max_iter=epoch)

            tr_pred = cnn.predict(self.training_data_X[:1000])
            ve_pred = cnn.predict(self.validation_data_X[:1000])

            tr_err_rate.append(np.mean(tr_pred != self.training_data_y[:1000]))
            ve_err_rate.append(np.mean(ve_pred != self.validation_data_y[:1000]))

            print(tr_pred)
        # Plot training and validation error rates

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, tr_err_rate, marker='o', label='Training Error')
        plt.plot(epochs, ve_err_rate, marker='x', linestyle='dashed', label='Validation Error')

        plt.xlabel("Number of Epochs")
        plt.ylabel("Error Rate")
        plt.title("Training & Validation Error vs. Number of Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()
        print(tr_err_rate)
        print(ve_err_rate)
        return tr_err_rate, ve_err_rate

    
    def train_learning_rate(self):
        learning_rates = [0.001, 0.01, 0.025, 0.05, 0.075, 0.1]
        
        tr_err_rate = []
        ve_err_rate = []

        for learning_rate in learning_rates:
            cnn = CNN()
            cnn.fit(self.training_data_X, self.training_data_y, learning_rate_init=learning_rate)

            tr_pred = cnn.predict(self.training_data_X)
            ve_pred = cnn.predict(self.validation_data_X)

            tr_err_rate.append(np.mean(tr_pred != self.training_data_y))
            ve_err_rate.append(np.mean(ve_pred != self.validation_data_y))
        
        plt.figure(figsize=(8, 5))
        plt.plot(learning_rates, tr_err_rate, marker='o', label='Training Error')
        plt.plot(learning_rates, ve_err_rate, marker='x', linestyle='dashed', label='Validation Error')

        plt.xscale("log")  # Log scale for better visualization
        plt.xlabel("Learning Rate")
        plt.ylabel("Error Rate")
        plt.title("Training & Validation Error vs. Learning Rate")
        plt.legend()
        plt.grid(True)
        plt.show()


    # MOST EXPENSIVE
    def train_optimally(self):
        learning_rates = [0.001, 0.01, 0.025, 0.05, 0.075, 0.1]
        epochs = [1, 10, 50, 75, 100, 125]
        
    
        results = {}

        for epoch in epochs:
            tr_err_rate = []
            ve_err_rate = []

            for learning_rate in learning_rates:

                cnn = CNN()
                cnn.fit(self.training_data_X, self.training_data_y, max_iter=epoch, learning_rate_init=learning_rate)

                tr_pred = cnn.predict(self.training_data_X)
                ve_pred = cnn.predict(self.validation_data_X)

                tr_err_rate.append(np.mean(tr_pred != self.training_data_y))
                ve_err_rate.append(np.mean(ve_pred != self.validation_data_y))
        
        results[epoch] = (tr_err_rate, ve_err_rate)

        plt.figure(figsize=(10, 6))

        for epoch in epochs:
            plt.plot(learning_rates, results[epoch][0], marker='o', label=f"Train Error (Epochs={epoch})")
            plt.plot(learning_rates, results[epoch][1], marker='x', linestyle='dashed', label=f"Validation Error (Epochs={epoch})")

        plt.xscale("log")  # Use log scale for better visualization
        plt.xlabel("Learning Rate")
        plt.ylabel("Error Rate")
        plt.title("Training & Validation Error vs. Learning Rate")
        plt.legend()
        plt.grid(True)
        plt.show()
            