import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import sys
# sys.path.append('/kaggle/input/nn-implementation')  # or '/kaggle/working' if that's where base.py is

# from base import NeuralNetwork
# from metrics import precision, recall

# Load UCI HAR dataset
X = pd.read_csv('/kaggle/input/uci-har-dataset/UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None)
y = pd.read_csv('/kaggle/input/uci-har-dataset/UCI HAR Dataset/train/y_train.txt', header=None)[0] - 1  # Zero-indexed

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create NN
nn = NeuralNetwork(input_dim=X_train.shape[1], hidden_dim=64, output_dim=len(np.unique(y)), learning_rate=0.01)

# Convert to numpy arrays
X_train_np = X_train
y_train_np = y_train.values
X_test_np = X_test
y_test_np = y_test.values

print(" Training started...")
nn.fit(X_train_np, y_train_np, epochs=20, batch_size=64)
print(" Training completed.")

print(" Prediction started...")
y_pred = nn.predict(X_test_np)
print(" Prediction completed.")

# Evaluate
print("\n Scratch Neural Network Performance:")
print(f"Accuracy: {accuracy(y_pred, y_test_np):.4f}")

for cls in np.unique(y_test_np):
    p = precision(y_pred, y_test_np, cls)
    r = recall(y_pred, y_test_np, cls)
    print(f"Class {cls+1}: Precision = {p:.3f}, Recall = {r:.3f}")
