import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred, cls):
    tp = np.sum((y_pred == cls) & (y_true == cls))
    fp = np.sum((y_pred == cls) & (y_true != cls))
    return tp / (tp + fp + 1e-9)

def recall(y_true, y_pred, cls):
    tp = np.sum((y_pred == cls) & (y_true == cls))
    fn = np.sum((y_pred != cls) & (y_true == cls))
    return tp / (tp + fn + 1e-9)

def f1_score(y_pred, y_true, cls=None):
    p = precision(y_pred, y_true, cls)
    r = recall(y_pred, y_true, cls)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0