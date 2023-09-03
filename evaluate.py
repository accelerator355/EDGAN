import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

def anomaly_scoring(x, re_x):
    scores = []
    for v1, v2 in zip(x, re_x):
        scores.append(np.sqrt(np.sum((v1 - v2) ** 2)))
    return np.array(scores)

def find_best(labels, scores, step = 2000):
    min_score = min(scores)
    max_score = max(scores)
    best_f1 = 0.0
    best_labels = None
    best_th = min_score
    for th in np.linspace(min_score, max_score, step):
        detected = (scores > th).astype(int)
        f1 = f1_score(y_true = labels, y_pred = detected)
        if f1 > best_f1:
            best_th = th
            best_labels = detected
            best_f1 = f1
    return best_th, best_labels, best_f1

def metrics_calculating(x, re_x, labels):
    scores = anomaly_scoring(x, re_x)
    best_th, best_labels, best_f1 = find_best(labels, scores)

    best_confusion_matrix = confusion_matrix(y_true = labels, y_pred = best_labels)
    TN = best_confusion_matrix[0, 0]
    FP = best_confusion_matrix[0, 1]
    FN = best_confusion_matrix[1, 0]
    TP = best_confusion_matrix[1, 1]

    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = precision_score(y_true = labels, y_pred = best_labels)
    recall = recall_score(y_true = labels, y_pred = best_labels)
    f1 = f1_score(y_true = labels, y_pred = best_labels)
    auc = roc_auc_score(y_true = labels, y_score = (scores - scores.min()) / (scores.max() - scores.min()))
    dice = ((2 * TP / (2 * TP + FN + FP)) +  (2 * TN / (2 * TN + FN + FP))) / 2
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FN) * (TP + FP) * (TN + FN) * (TN + FP))

    return {'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'dice': dice,
            'mcc': mcc}




