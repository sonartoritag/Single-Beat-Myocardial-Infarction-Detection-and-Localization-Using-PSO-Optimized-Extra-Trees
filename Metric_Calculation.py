import numpy as np
def metrics(conf_mat):
    specificities = []
    for i in range(conf_mat.shape[0]):
        # True Negatives (TN) are all elements except those in the ith row and ith column
        TN = conf_mat.sum() - conf_mat[i, :].sum() - conf_mat[:, i].sum() + conf_mat[i, i]
        # False Positives (FP) are the sum of the column, excluding the diagonal
        FP = conf_mat[:, i].sum() - conf_mat[i, i]
        specificity = TN / (TN + FP)
        specificities.append(specificity)

    for i, specificity in enumerate(specificities):
        print(f"Specificity for class {i}: {specificity:.4f}")
    print(np.mean(specificities))

    sensitivities = []
    # For each class (0 to 6 in this case), calculate sensitivity
    for i in range(conf_mat.shape[0]):
        # True Positives (TP) are the diagonal elements
        TP = conf_mat[i, i]
        # False Negatives (FN) are the sum of the ith row, excluding the diagonal
        FN = conf_mat[i, :].sum() - TP
        sensitivity = TP / (TP + FN)
        sensitivities.append(sensitivity)

    # Print sensitivity for each class
    for i, sensitivity in enumerate(sensitivities):
        print(f"Sensitivity for class {i}: {sensitivity:.4f}")
    print(np.mean(sensitivities))


    accuracies = []
    total_samples = conf_mat.sum()
    for i in range(conf_mat.shape[0]):
        # True Positives (TP) for class i are the diagonal elements
        TP = conf_mat[i, i]
        # True Negatives (TN) are the sum of all elements except those in the i-th row and i-th column
        TN = conf_mat.sum() - conf_mat[i, :].sum() - conf_mat[:, i].sum() + conf_mat[i, i]
        accuracy = (TP + TN) / total_samples
        accuracies.append(accuracy)

    # Print accuracy for each class
    for i, accuracy in enumerate(accuracies):
        print(f"Accuracy for class {i}: {accuracy:.4f}")
    print(np.mean(accuracies))

    f1_scores = []
    for i in range(conf_mat.shape[0]):
        # True Positives (TP) for class i are the diagonal elements
        TP = conf_mat[i, i]
        # False Positives (FP) are the sum of the i-th column, excluding the diagonal
        FP = conf_mat[:, i].sum() - TP
        # False Negatives (FN) are the sum of the i-th row, excluding the diagonal
        FN = conf_mat[i, :].sum() - TP
        # Precision for class i
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        # Recall for class i (same as sensitivity)
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        # F1 score for class i
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        f1_scores.append(f1)

    # Print F1 score for each class
    for i, f1 in enumerate(f1_scores):
        print(f"F1 score for class {i}: {f1:.4f}")
    print(np.mean(f1_scores))

    mcc_scores = []
    # For each class (0 to 6 in this case), calculate MCC
    for i in range(conf_mat.shape[0]):
        # True Positives (TP) for class i
        TP = conf_mat[i, i]
        # False Positives (FP) for class i
        FP = conf_mat[:, i].sum() - TP
        # False Negatives (FN) for class i
        FN = conf_mat[i, :].sum() - TP
        # True Negatives (TN) for class i
        TN = conf_mat.sum() - (TP + FP + FN)

        # Calculate MCC for class i
        numerator = TP * TN - FP * FN
        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

        mcc = numerator / denominator if denominator != 0 else 0  # Prevent division by zero
        mcc_scores.append(mcc)

    # Print MCC for each class
    for i, mcc in enumerate(mcc_scores):
        print(f"MCC for class {i}: {mcc:.4f}")
    print(np.mean(mcc_scores))


def get_metrics(conf_mat):
    num_classes = conf_mat.shape[0]
    total = conf_mat.sum()

    sensitivity_list = []
    specificity_list = []
    f1_list = []
    mcc_list = []
    accuracy_list = []

    for i in range(num_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        TN = total - (TP + FP + FN)

        # Accuracy for this class
        acc = (TP + TN) / total if total != 0 else 0
        accuracy_list.append(acc)

        # Sensitivity (Recall)
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        sensitivity_list.append(sensitivity)

        # Specificity
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        specificity_list.append(specificity)

        # Precision
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0

        # F1 Score
        f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
        f1_list.append(f1)

        # Matthews Correlation Coefficient
        numerator = (TP * TN) - (FP * FN)
        denominator = np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
        mcc = numerator / denominator if denominator != 0 else 0
        mcc_list.append(mcc)

    # Average of all metrics
    print('Average Accuracy', np.mean(accuracy_list))
    print('Average Sensitivity', np.mean(sensitivity_list))
    print('Average Specificity', np.mean(specificity_list))
    print('Average F1 Score', np.mean(f1_list))
    print('Average MCC', np.mean(mcc_list))

