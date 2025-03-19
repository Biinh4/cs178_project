import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import numpy as np

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
        mlp_sgd = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),  # 3 hidden layers
                activation="relu",  # Activation function
                solver="sgd",  # Use Stochastic Gradient Descent
                learning_rate="adaptive",  # Adjusts learning rate dynamically
                learning_rate_init=0.01,  # Initial learning rate
                momentum=0.9,  # Helps stabilize updates
                max_iter=500,  # More iterations since SGD is noisier
                random_state=42
            ))
        ])

        # Train the model
        mlp_sgd.fit(self.X_train, self.y_train)

        # Predictions
        y_pred_sgd = mlp_sgd.predict(self.X_test)

        # Accuracy
        accuracy_sgd = accuracy_score(self.y_test, y_pred_sgd)
        print(f"Test Accuracy (SGD): {accuracy_sgd:.4f}")

    
    def train_learning_rate(self):
        learning_rates = [0.001, 0.025, 0.05, 0.1, 0.25, 0.5]

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
        layers = [(64, 32, 16), (32, 16, 4), (128, )]

        training_errors = []
        testing_errors = []

        for layer_config in layers:
            mlp_sgd = Pipeline([
                ("preprocessor", self.preprocessor),
                ("classifier", MLPClassifier(
                    hidden_layer_sizes=layer_config,  # Layer configuration
                    activation="relu",  # Activation function
                    solver="sgd",  # Use Stochastic Gradient Descent
                    learning_rate="adaptive",  # Adjusts learning rate dynamically
                    learning_rate_init=0.01,  # Initial learning rate
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
        
        # Y
        learning_rates = [0.001, 0.025, 0.05, 0.1, 0.25, 0.5]

        # X
        layers = [(64, 32, 16), (32, 16, 4), (128, )]

        training_errors = []
        testing_errors = []

        for layer_config in layers:
            for learning_rate in learning_rates:
                mlp_sgd = Pipeline([
                    ("preprocessor", self.preprocessor),
                    ("classifier", MLPClassifier(
                        hidden_layer_sizes=layer_config,  # Layer configuration
                        activation="relu",  # Activation function
                        solver="sgd",  # Use Stochastic Gradient Descent
                        learning_rate=learning_rate,  # Adjusts learning rate dynamically
                        learning_rate_init=0.01,  # Initial learning rate
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
