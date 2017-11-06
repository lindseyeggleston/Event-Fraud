import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve
from scipy import interp
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_ROC_curve(classifiers, X, y, benchmarks=None, balancing=[], pos_label=1, n_folds=5, save_path=None):
    '''
    Input:
    -classifiers is a list of sklearn classifier objects
    -balancing is a list of sklearn over- and undersampling objects

    Output:
    -a single plot with ROC curves for all balancing-classifier combinations
    '''

    plt.close('all')
    fig, ax = plt.subplots(figsize=(8,8))
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
    if len(balancing) > 0:
        print('Preprocessing...')
        for cl in classifiers:
                for b in balancing:
                    mean_tpr, mean_fpr, mean_auc = _get_ROC_curve(cl, X, y, b)
                    ax.plot(mean_fpr, mean_tpr, label=cl.__class__.__name__ + ' (AUC = %0.3f)' % mean_auc, lw=2, zorder=1)

    else:
        for cl in classifiers:
            mean_tpr, mean_fpr, mean_auc = _get_ROC_curve(cl, X, y)
            ax.plot(mean_fpr, mean_tpr, label=cl.__class__.__name__ + ' (AUC = %0.3f)' % mean_auc, lw=2, zorder=1)

    ax.plot([0, 1], [0, 1], '--', color='black', label='Random')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15

    ax.set_xlabel('FPR (n = {})'.format(len(y) - y.sum()), fontsize=18)
    ax.set_ylabel('TPR (n = {})'.format(y.sum()), fontsize=18)
    # plt.title('ROC curve')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, dpi=600, transparent=False)
    else:
        plt.show()

def _get_ROC_curve(classifier, X, y, balancing=None, pos_label=1, n_folds=5):
    '''
    Called by plot_ROC_curve.  Outputs mean ROC curve and mean AUC of all folds in n-fold validation.
    '''
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 400)
    all_tpr = []
    skf = StratifiedKFold(n_splits=n_folds, random_state=40, shuffle=True)
    for train, test in skf.split(X, y):
        X_train, y_train = X.iloc[train], y[train]
        if balancing:
            X_train, y_train = balancing.fit_sample(X_train, y_train)
        classifier.fit(X_train, y_train)
        probas_ = classifier.predict_proba(X.iloc[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1], pos_label)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        # plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= n_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    return mean_tpr, mean_fpr, mean_auc

if __name__ == '__main__':
    print('Please import this script to call its functions.')
