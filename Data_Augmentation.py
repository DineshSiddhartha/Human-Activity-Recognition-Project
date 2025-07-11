# !pip install augly


import pandas as pd
import numpy as np
import augly.audio as aud
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------
# Load your original UCI HAR data
X_orig = pd.read_csv('/kaggle/input/uci-har-dataset/UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None)
y_orig = pd.read_csv('/kaggle/input/uci-har-dataset/UCI HAR Dataset/train/y_train.txt', header=None)[0]

# Use small subset for demo (can remove later)
# X_orig = X_raw
# y_orig = y_true

# -----------------------------------------------------------------
# Function to simulate signal augmentation using Augly
def augment_sample_row(row):
    sig = row[:500].values.astype(np.float32)
    sig = np.nan_to_num(sig)

    # Apply augmentation
    aug = aud.AddBackgroundNoise(snr_level_db=10)
    sig_aug = aug(sig)  # In latest augly, returns np.array directly; if error persists, check below

    # If it returns tuple: sig_aug, _ = aug(sig)
    if isinstance(sig_aug, tuple):
        sig_aug, _ = sig_aug

    # Compute simple features on augmented signal
    feat_mean = np.mean(sig_aug)
    feat_std = np.std(sig_aug)
    feat_energy = np.sum(sig_aug ** 2)

    return [feat_mean, feat_std, feat_energy]


# -----------------------------------------------------------------
# Build augmented features
augmented_features = []

print("Creating augmented features...")

for i in range(len(X_orig)):
    f = augment_sample_row(X_orig.iloc[i])
    augmented_features.append(f)

augmented_features = np.array(augmented_features)