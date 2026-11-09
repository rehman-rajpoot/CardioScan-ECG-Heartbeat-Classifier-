import numpy as np

def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

def confusion_matrix(y_true, y_pred, num_classes=5):
    C = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    return C

def per_class_accuracy(C):
    accs = []
    for j in range(len(C)):
        row_total = C[j].sum()
        accs.append(float(C[j, j] / row_total))
    return np.array(accs)

def balanced_accuracy(C):
    return float(np.mean(per_class_accuracy(C)))
