# 🏃‍♂️ Human Activity Recognition using Wearable Sensor Data

## Project Overview

This project tackles the challenge of **human activity recognition (HAR)** using smartphone-based accelerometer data from the UCI HAR dataset. The goal is to classify activities such as walking, sitting, standing, and more, using robust signal processing and machine learning models.

---

## Dataset

- **UCI HAR Dataset**: Contains accelerometer and gyroscope signals from 30 participants performing 6 daily activities.
- **Self-collected samples**: Additional accelerometer data recorded to improve model generalization and robustness.

---

## Data Preparation

- Combined raw inertial signals (`total_acc_x`, `total_acc_y`, `total_acc_z`) into subject-activity-specific CSV files.
- Applied **offset-based slicing** and **windowing** to create consistent, noise-resistant signal segments.
- Created a **stratified train-test split** to ensure balanced representation across activity classes.

---

## Feature Engineering

- Used **TSFEL** (Time Series Feature Extraction Library) to automatically extract a comprehensive set of time-series features.
- Performed additional analysis and transformation to enhance model readiness.

---

## Exploratory Data Analysis (EDA)

- Visualized raw and windowed accelerometer signals using Matplotlib and Seaborn.
- Identified activity-specific patterns and noise characteristics.

---

## Classical Machine Learning

- **Decision Tree Classifier (Scikit-learn)**: Achieved ~94% accuracy.
- **Custom Decision Tree (from scratch)**: Implemented using NumPy and Pandas, reaching ~84% accuracy.

---

## Deep Learning Models

- **PyTorch Neural Network**: Designed and trained a neural network, achieving ~93% accuracy. (Architecture included fully connected or convolutional layers — depending on the final implementation.)
- **Scratch Neural Network**: Built a simplified neural network implementation from the ground up, achieving similar performance.

---

## Data Augmentation

- Used **AugLy** library to augment signals (e.g., noise addition, temporal distortions) and improve generalization.
- Included **self-collected accelerometer data** to enhance robustness against real-world variations.
- Final accuracy on mixed and augmented samples: **~95%**.

---

## Validation & Refinement

- Tested on new, real-world collected samples.
- Refined models using iterative augmentation and mixed-sample training to boost reliability.

---

## Highlights

- End-to-end pipeline: data engineering ➡️ feature extraction ➡️ classical ML ➡️ deep learning ➡️ augmentation ➡️ real-world validation.
- Custom algorithm implementations (decision tree and neural net) to deepen understanding.
- Focus on reproducibility, interpretability, and practical deployment readiness.

---

## Future Directions

- Deploy the final model to mobile or embedded devices.
- Add additional activity classes or fine-grained movements.
- Visualize confusion matrices and misclassifications in detail.
- Explore advanced sequence models like RNNs or Transformers.

---

## Acknowledgements

- **UCI HAR Dataset** for providing a comprehensive benchmark dataset.
- Libraries used: NumPy, Pandas, Matplotlib, Seaborn, TSFEL, Scikit-learn, PyTorch, AugLy.

---

**If you'd like to discuss this project or collaborate on wearable sensing and activity recognition, feel free to connect!**

