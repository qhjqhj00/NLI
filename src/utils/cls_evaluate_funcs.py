
import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score, recall_score, precision_score



def cal_f1_score(pcs, rec):
    tmp = 2 * pcs * rec / (pcs + rec)
    return round(tmp, 4)



def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    if len(set(labels)) > 2:
        average = 'micro'
    else:
        average = 'binary'
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    recall = recall_score(y_true=labels, y_pred=preds, average=average)
    precision = precision_score(y_true=labels, y_pred=preds, average=average)
    return {
        "acc": acc,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "acc_and_f1": (acc + f1) / 2,
    }


def f1_measures(preds, labels):
    if len(set(labels)) > 2:
        average = 'micro'
    else:
        average = 'binary'
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    recall = recall_score(y_true=labels, y_pred=preds, average=average)
    precision = precision_score(y_true=labels, y_pred=preds, average=average)
    return {
        "f1": f1,
        "recall": recall,
        "precision": precision
    }


def acc(preds, labels):
    acc = simple_accuracy(preds, labels)
    return {
        "acc": acc,
    }
