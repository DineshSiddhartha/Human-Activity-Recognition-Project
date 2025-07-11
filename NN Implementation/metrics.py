# =====================
# Metric functions
# =====================
import numpy as np

def precision(y_true, y_pred, cls):
    y_true_cls = (y_true == cls)
    y_pred_cls = (y_pred == cls)
    tp = np.sum(y_true_cls & y_pred_cls)
    fp = np.sum(~y_true_cls & y_pred_cls)
    return tp / (tp + fp + 1e-9)

def recall(y_true, y_pred, cls):
    y_true_cls = (y_true == cls)
    y_pred_cls = (y_pred == cls)
    tp = np.sum(y_true_cls & y_pred_cls)
    fn = np.sum(y_true_cls & ~y_pred_cls)
    return tp / (tp + fn + 1e-9)

def f1_score(y_true, y_pred, cls):
    p = precision(y_true, y_pred, cls)
    r = recall(y_true, y_pred, cls)
    return 2 * p * r / (p + r + 1e-9)

def macro_f1(y_true, y_pred):
    classes = np.unique(y_true)
    f1s = [f1_score(y_true, y_pred, cls) for cls in classes]
    return np.mean(f1s)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def one_hot(y, num_classes=None):
    if num_classes is None:
        num_classes = np.max(y) + 1
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out