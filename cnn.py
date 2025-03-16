import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import scipy.io
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


class CNN:
    def __init__(self):
        # Initialize parameters: assumes data size! 28x28 and 10 classes
        self.conv_ = torch.nn.Conv2d(3, 16, (5,5), stride=2)  # Be careful when declaring sizes;
        self.pool_ = torch.nn.MaxPool2d(3, stride=2)          # inconsistent sizes will give you
        self.lin_ = torch.nn.Linear(576,10)                   # hard-to-read error messages.
            
    def forward_(self,X):
        """Compute NN forward pass and output class probabilities (as tensor) """
        r1 = self.conv_(X)             # X is (m,1,28,28); R is (m,16,24,24)/2 = (m,16,12,12)
        h1 = torch.relu(r1)            #
        h1_pooled = self.pool_(h1)     # H1 is (m,16,12,12), so H1p is (m,16,10,10)/2 = (m,16,5,5)
        h1_flat = torch.nn.Flatten()(h1_pooled)  # and H1f is (m,400)
        r2 = self.lin_(h1_flat)
        f  = torch.softmax(r2, dim=1)  # Output is (m,10)
        return f 
        
    def parameters(self): 
        return list(self.conv_.parameters())+list(self.pool_.parameters())+list(self.lin_.parameters())
    
    def predict(self,X):
        """Compute NN class predictions (as array) """
        # m = X.shape[3]
        # n = X.shape[0] * X.shape[1] * X.shape[2]

        # Xtorch = torch.tensor(X).reshape(m,1,int(np.sqrt(n)),int(np.sqrt(n)))
        # return self.classes_[np.argmax(self.forward_(Xtorch).detach().numpy(),axis=1)]   # pick the most probable class

        # Had to change the order of the X
        Xtorch = torch.tensor(X).permute(0, 3, 1, 2).float()  # (32,32,3,26032) → (26032,3,32,32)
        return self.classes_[np.argmax(self.forward_(Xtorch).detach().numpy(), axis=1)]

    def J01(self,X,y):   return (y != self.predict(X)).mean()

    def JNLL_(self,X,y): 
        # Converts our classes from 1-10 to 0-9
        y = torch.tensor(y).long() - 1
        return -torch.log(self.forward_(X)[range(len(y)),y.long()]).mean()

    def fit(self, X,y, batch_size=256, max_iter=100, learning_rate_init=.005, momentum=0.9, alpha=.001, plot=False):
        self.classes_ = np.unique(y)        
        # m = X.shape[3]
        # n = X.shape[0] * X.shape[1] * X.shape[2]
        
        # Xtorch = torch.tensor(X).reshape(m,1,int(np.sqrt(n)),int(np.sqrt(n)))  
        # self.loss01, self.lossNLL = [self.J01(X,y)], [float(self.JNLL_(Xtorch,y))]

        # Reshaping our data
        Xtorch = torch.tensor(X).permute(0, 3, 1, 2).float()
        # print(Xtorch.shape)
        ytorch = torch.tensor(y, dtype=torch.long)

        self.loss01, self.lossNLL = [self.J01(X, y)], [float(self.JNLL_(Xtorch, ytorch))]

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate_init)

        m = X.shape[0]
        for epoch in range(max_iter):                        # 1 epoch = pass through all data
            pi = np.random.permutation(m)                    # per epoch: permute data order
            for ii,i in enumerate(range(0,m,batch_size)):    # & split into mini-batches
                ivals = pi[i:i+batch_size]
                optimizer.zero_grad()                        # Reset the gradient computation
                Ji = self.JNLL_(Xtorch[ivals,:,:,:],y[ivals])
                Ji.backward()
                optimizer.step()
            self.loss01.append(self.J01(X,y))                # track 0/1 and NLL losses
            self.lossNLL.append(float(self.JNLL_(Xtorch,y)))
            
            if plot:                                         # optionally visualize progress
                display.clear_output(wait=True)
                plt.plot(range(epoch+2),self.loss01,'b-',range(epoch+2),self.lossNLL,'c-')
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


    def train(self):
        cnn = CNN()
        cnn.fit(self.training_data_X[:250], self.training_data_y[:250])
