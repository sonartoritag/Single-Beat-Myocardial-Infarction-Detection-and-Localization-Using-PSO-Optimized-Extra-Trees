import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import precision_recall_curve, average_precision_score


def ROC_cur(y_test, y_scores):
    y_test_flat = y_test.ravel()
    class_labels = ['1', '21', '22', '23', '24', '25', '26']
    class_names = ['HC', 'A', 'AS', 'AL', 'I', 'IL', 'IPL']

    fpr = {}
    tpr = {}
    roc_auc = {}

    # Convert string labels to numeric for comparison
    y_test_int = np.array([int(label) for label in y_test_flat])

    plt.figure(figsize=(10, 8))

    for idx, class_label in enumerate(class_labels):
        class_int = int(class_label)
        y_true_binary = (y_test_int == class_int).astype(int)
        fpr[idx], tpr[idx], _ = roc_curve(y_true_binary, y_scores[:, idx])
        roc_auc[idx] = auc(fpr[idx], tpr[idx])
        plt.plot(fpr[idx], tpr[idx], lw=6, label=f'Class {class_names[idx]} ROC (AUC = {roc_auc[idx]:.2f})')

    # Plot the diagonal (random guess)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)

    # Labels and title
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title(r'ROC Curves for HC and 6 MI classes of $\mathit{inp}$Mdl#2', fontsize=20)
    plt.legend(loc='lower right', fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.show()

def PRC_cur(y_test, y_scores):
    y_test_flat = y_test.ravel()
    class_labels = ['1', '21', '22', '23', '24', '25', '26']
    class_names = ['HC', 'A', 'AS', 'AL', 'I', 'IL', 'IPL']

    precision = {}
    recall = {}
    avg_precision = {}

    # Convert string labels to integers if needed
    y_test_int = np.array([int(label) for label in y_test_flat])

    plt.figure(figsize=(10, 8))

    for idx, class_label in enumerate(class_labels):
        class_int = int(class_label)
        y_true_binary = (y_test_int == class_int).astype(int)
        precision[idx], recall[idx], _ = precision_recall_curve(y_true_binary, y_scores[:, idx])
        avg_precision[idx] = average_precision_score(y_true_binary, y_scores[:, idx])
        plt.plot(recall[idx], precision[idx], lw=6, label=f'Class {class_names[idx]} PRC (Average Precision = {avg_precision[idx]:.2f})')

    # Labels and title
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.title(r'Precision-Recall Curves for HC and 6 MI classes of $\mathit{inp}$Mdl#2', fontsize=20)
    plt.legend(loc='lower left', fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.show()