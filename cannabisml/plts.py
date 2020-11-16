"""Plotting functions."""

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt


def plot_perfomance(probas_list, y_test_list):
    """Plot AUC of classifier performance

    Parameters
    ----------
    probas_list : list
        list of prediction probabilities for each outer fold
    y_test_list : list
        list of labels in each out fold test split
    """
    # Results visulazation
    plt.figure(figsize=(10,10), dpi=500)

    # ROC for each fold
    tprs = []
    aucs = []
    for idx, probas in enumerate(probas_list):
        mean_fpr = np.linspace(0, 1, 100)
        fpr, tpr, thresholds = roc_curve(y_test_list[idx], probas[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        loop = idx + 1
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC Fold %d (AUC = %0.2f)' % (loop, roc_auc))

    # Mean ROC and confidence interval
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 Standard Deviation')

    # Chance
    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")

    # Graph Labels
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
