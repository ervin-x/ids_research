import  numpy as np
from matplotlib import  pyplot as plt

from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def print_optimal_threshold(pred_proba, y_test):
    # Youden’s J statistic - determines the threshold value at which
    # the graph reaches the closest point to (0, 1) on the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba)
    J = tpr - fpr
    ix = np.argmax(J)
    print(f"Best threshold by Youden’s J statistic: {round(thresholds[ix], 3)}")

    # Determining threshold by maximizing the F-Score
    pr, rec, thresholds = precision_recall_curve(y_test, pred_proba)
    fscore = (2 * pr * rec) / (pr + rec)
    ix = np.argmax(fscore)
    print(
        f"Best threshold by optimising F-score: {round(thresholds[ix], 3)}, F-Score={round(fscore[ix], 3)}"
    )


def print_custom_classification_report(y_test, pred_proba, threshold):
    print(f"threshold:{threshold}")
    pred = pred_proba.copy()
    pred_proba = np.array(pred_proba)
    pred[pred_proba >= threshold] = 1
    pred[pred_proba < threshold] = 0
    print(classification_report(y_test, pred))


def print_corves(pred_proba, y_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Metrics")

    fpr, tpr, thresholds = roc_curve(y_test, pred_proba)
    ax1.plot([0, 1], [0, 1], linestyle="--", label="No Skill")
    ax1.plot(fpr, tpr, marker=".", label="Logistic")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend()
    ax1.set_title("ROC curve")

    pr, rec, thresholds = precision_recall_curve(y_test, pred_proba)
    ax2.plot(thresholds, pr[:-1], c="r", label="PRECISION")
    ax2.plot(thresholds, rec[:-1], c="b", label="RECALL")
    ax2.grid()
    ax2.legend()
    ax2.set_title("Precision-Recall Curve")
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot(ax=ax)
    plt.show()
