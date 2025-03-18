import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# Define column names
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
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# Split dataset
X = df.drop(columns=["income"])
y = df["income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define MLP Classifier with SGD
mlp_sgd = Pipeline([
    ("preprocessor", preprocessor),
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
mlp_sgd.fit(X_train, y_train)

# Predictions
y_pred_sgd = mlp_sgd.predict(X_test)

# Accuracy
accuracy_sgd = accuracy_score(y_test, y_pred_sgd)
print(f"Test Accuracy (SGD): {accuracy_sgd:.4f}")
