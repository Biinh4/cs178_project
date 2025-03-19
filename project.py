import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
import numpy as np
import time
import matplotlib.pyplot as plt


# Define column names

class train_nn:
    def __init__(self):
        columns = [
            "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
            "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
            "hours_per_week", "native_country", "income"
        ]

        # Load the dataset
        df = pd.read_csv("adult/adult.data", names=columns, na_values=" ?", skipinitialspace=True)

        # Drop missing values
        df.dropna(inplace=True)

        # Convert target variable to binary (0 = <=50K, 1 = >50K)
        df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

        # Identify categorical and numerical columns
        categorical_cols = ["workclass", "education", "marital_status", "occupation",
                            "relationship", "race", "sex", "native_country"]
        numerical_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

        # Preprocessing pipeline: one-hot encoding for categorical & standard scaling for numerical
        self.preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ])

        # Split dataset
        X = df.drop(columns=["income"])
        y = df["income"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    def basic_train(self):

        # Define MLP Classifier with SGD

        start_time = time.time()
        mlp_sgd = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),  # 3 hidden layers
                activation="relu",  # Activation function
                solver="sgd",  # Use Stochastic Gradient Descent
                learning_rate="adaptive",  # Adjusts learning rate dynamically
                learning_rate_init=0.001,  # Initial learning rate
                momentum=0.9,  # Helps stabilize updates
                max_iter=500,  # More iterations since SGD is noisier
                random_state=42
            ))
        ])

        # Train the model
        mlp_sgd.fit(self.X_train, self.y_train)

        # Predictions
        tr_y_pred_sgd = mlp_sgd.predict(self.X_train)
        ve_y_pred_sgd = mlp_sgd.predict(self.X_test)

        # Accuracy
        print(f"Training Error rate (SGD): {np.mean(tr_y_pred_sgd != self.y_train)}")
        print(f"Validation Error rate (SGD): {np.mean(ve_y_pred_sgd != self.y_test)}")

        end_time = time.time()
        elapsed_time = end_time - start_time  # Compute elapsed time

        print(f"Training took {elapsed_time:.2f} seconds")
        print("Most optimal layers: (64, 32, 16).")
        print("Most optimal learning rate: 0.001.")

    
    def train_learning_rate(self):
        learning_rates = [0.001, 0.0025, 0.005, 0.01, 0.015, 0.02]

        training_errors = []
        testing_errors = []

        for learning_rate in learning_rates:
            mlp_sgd = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),  # 3 hidden layers
                activation="relu",  # Activation function
                solver="sgd",  # Use Stochastic Gradient Descent
                learning_rate="adaptive",  # Adjusts learning rate dynamically
                learning_rate_init=learning_rate,  # Initial learning rate
                momentum=0.9,  # Helps stabilize updates
                max_iter=500,  # More iterations since SGD is noisier
                random_state=42
            ))
            ])

            # Train the model
            mlp_sgd.fit(self.X_train, self.y_train)

            # Predictions
            tr_y_pred_sgd = mlp_sgd.predict(self.X_train)
            tst_y_pred_sgd = mlp_sgd.predict(self.X_test)

            training_errors.append(np.mean(tr_y_pred_sgd != self.y_train))
            testing_errors.append(np.mean(tst_y_pred_sgd != self.y_test))
        
        return (learning_rates, training_errors), (learning_rates, testing_errors)


    def train_layers(self):
        layers = [(128, 64, 32), (64, 32, 16), (256, 128, 64), (100, 50, 25)]

        training_errors = []
        testing_errors = []

        for layer_config in layers:
            print(training_errors)
            mlp_sgd = Pipeline([
                ("preprocessor", self.preprocessor),
                ("classifier", MLPClassifier(
                    hidden_layer_sizes=layer_config,  # Layer configuration
                    activation="relu",  # Activation function
                    solver="sgd",  # Use Stochastic Gradient Descent
                    learning_rate="adaptive",  # Adjusts learning rate dynamically
                    learning_rate_init=0.001,  # Initial learning rate
                    momentum=0.9,  # Helps stabilize updates
                    max_iter=500,  # More iterations since SGD is noisier
                    random_state=42
                ))
            ])

            # Train the model
            mlp_sgd.fit(self.X_train, self.y_train)

            # Predictions
            tr_y_pred_sgd = mlp_sgd.predict(self.X_train)
            tst_y_pred_sgd = mlp_sgd.predict(self.X_test)

            training_errors.append(np.mean(tr_y_pred_sgd != self.y_train))
            testing_errors.append(np.mean(tst_y_pred_sgd != self.y_test))
        
        return (layers, training_errors), (layers, testing_errors)
    

    def train_optimal(self):
        learning_rates = [0.001, 0.0025, 0.005, 0.01, 0.015, 0.02]
        layers = [(128, 64, 32), (64, 32, 16), (256, 128, 64), (100, 50, 25)]

        training_errors = {}
        testing_errors = {}

        # Preprocess once

        for layer_config in layers:
            training_errors[layer_config] = []
            testing_errors[layer_config] = []

            for learning_rate in learning_rates:
                mlp_sgd = Pipeline([
                ("preprocessor", self.preprocessor),
                ("classifier", MLPClassifier(
                    hidden_layer_sizes=layer_config,  # Layer configuration
                    activation="relu",  # Activation function
                    solver="sgd",  # Use Stochastic Gradient Descent
                    learning_rate="adaptive",  # Adjusts learning rate dynamically
                    learning_rate_init=learning_rate,  # Initial learning rate
                    momentum=0.9,  # Helps stabilize updates
                    max_iter=500,  # More iterations since SGD is noisier
                    random_state=42
                ))
                ])

                # Train the model
                mlp_sgd.fit(self.X_train, self.y_train)

                # Predictions
                tr_y_pred_sgd = mlp_sgd.predict(self.X_train)
                tst_y_pred_sgd = mlp_sgd.predict(self.X_test)

                # Calculate errors
                train_error = np.mean(tr_y_pred_sgd != self.y_train)
                test_error = np.mean(tst_y_pred_sgd != self.y_test)

                # Store results
                training_errors[layer_config].append(train_error)
                testing_errors[layer_config].append(test_error)

        return training_errors, testing_errors
        

    def repesent_learning_rate(self):
        training, testing = self.train_learning_rate()

        plt.figure(figsize=(8, 5))
        plt.plot(training[0], training[1], marker='o', linestyle='-', color='b', label="Training Error")
        plt.plot(testing[0], testing[1], marker='s', linestyle='-', color='r', label="Testing Error")
        plt.xscale("log")  # Log scale for better visualization
        plt.xlabel("Learning Rate")
        plt.ylabel("Error Rate")
        plt.title("Training vs Testing Error for Different Learning Rates")
        plt.legend()
        plt.grid()
        plt.show()


    def represent_layers(self):
        training, testing = self.train_layers()

        layer_configs, training_errors = training
        _, testing_errors = testing

        # Convert tuples to strings for categorical x-axis labels
        x_labels = [str(layer) for layer in layer_configs]

        # Create x-axis indices
        x_indices = np.arange(len(x_labels))  # Numeric positions

        # Plot Training and Testing Errors
        plt.figure(figsize=(8, 6))
        plt.plot(x_indices, training_errors, 'bo-', label="Training Error")
        plt.plot(x_indices, testing_errors, 'rs-', label="Testing Error")

        # Set x-ticks to the string labels
        plt.xticks(x_indices, x_labels, rotation=30, ha="right")

        # Labeling
        plt.xlabel("Layer Configuration (Hidden Layers)")
        plt.ylabel("Error Rate")
        plt.title("Training vs Testing Error for Different Layer Configurations")
        plt.legend()
        plt.show()


    def represent_optimal(self):
        training_errors, testing_errors = self.train_optimal()
        learning_rates = [0.001, 0.0025, 0.005, 0.01, 0.015, 0.02]
        layers = [(128, 64, 32), (64, 32, 16), (256, 128, 64), (100, 50, 25)]

        x_labels = [str(layer) for layer in layers]
        x_indices = np.arange(len(layers))

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot Training Errors
        for i, lr in enumerate(learning_rates):
            ax.scatter(x_indices, [lr] * len(layers), training_errors[:, i], label=f"LR {lr} (Train)", marker='o')

        # Plot Testing Errors
        for i, lr in enumerate(learning_rates):
            ax.scatter(x_indices, [lr] * len(layers), testing_errors[:, i], label=f"LR {lr} (Test)", marker='s')

        # Labels
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, rotation=20)
        ax.set_xlabel("Layers")
        ax.set_ylabel("Learning Rate")
        ax.set_zlabel("Error Rate")
        ax.set_title("Error Rate vs Learning Rate and Layers")

        plt.legend()
        plt.show()