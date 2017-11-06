import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.metrics import confusion_matrix

def get_cost_benefit(revenue, cost):
    '''
    Takes two integers: Revenue of correctly identified fraud, and cost of investigating any posting flagged for review.
    Returns cost-benefit matrix
    '''
    benefit = revenue - cost
    cost = -cost
    return np.array([[benefit, cost],[0, 0]])

def standard_confusion_matrix(y_true, y_pred):
    '''
    Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    '''

    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])

def profit_curve(predicted_probs, labels, revenue, cost, thresholds):
    '''
    Calculates a list of profits based on:
    1) cost-benefit matrix;
    2) predicted probabilities of data points;
    3) the true labels.

    Parameters
    ----------
    predicted_probs : ndarray - 1D, predicted probability for each datapoint
                                    in labels, in range [0, 1]
    labels          : ndarray - 1D, true label of datapoints, 0 or 1

    Returns
    -------
    profits    : ndarray - 1D
    thresholds : ndarray - 1D

    Based on a profit curve function in Galvanize g39ds-soln.
    '''

    actual_positives = float(sum(labels))
    cost_benefit = get_cost_benefit(revenue, cost)
    profits = []
    pred_positive_rates = []
    for threshold in thresholds:
        y_predict = np.array([1 if p >= threshold else 0 for p in predicted_probs])
        confusion_matrix = standard_confusion_matrix(labels, y_predict)
        threshold_profit = np.sum(confusion_matrix * cost_benefit)/actual_positives
        # print(confusion_matrix)
        # print(threshold_profit)
        # print('----------------')
        profits.append(threshold_profit)
    return np.array(profits)

def plot_avg_profits(classifier, X, y, n_splits=5, revenue=75, cost=25, save_path=None):
    '''
    Plots the average profit curve across n-fold validation of a model
    '''

    plt.close('all')
    fig = plt.figure(figsize=(9,6))
    plt.rcParams.update({'font.size': 14})
    ax1 = fig.add_subplot(1,1,1)
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    skf = StratifiedKFold(n_splits=n_splits, random_state=40, shuffle=True)
    thresholds = np.linspace(.05, 1, 200)
    avg_profits = np.zeros(len(thresholds))
    print('Predicting...')
    for train, test in skf.split(X, y):
        classifier.fit(X.iloc[train], y[train])
        predicted_probs = classifier.predict_proba(X.iloc[test])[:,1]
        profits = profit_curve(predicted_probs, y[test], revenue, cost, thresholds)
        avg_profits += profits
        # plot_model_profits(y[test], predicted_probs, revenue, cost, ax1)
    avg_profits = avg_profits/n_splits
    max_idx = avg_profits.argmax()
    max_profit = int(avg_profits[max_idx])
    ax1.scatter(thresholds[max_idx], max_profit, color='yellow', zorder=2) #max profit point
    ax1.annotate('Maximum profit: ${}'.format(int(max_profit)), (thresholds[max_idx], max_profit), xytext=(8, 3), textcoords='offset points', fontsize=14)
    ax1.plot(thresholds, avg_profits, color='blue', linewidth=1.4, zorder=1)
    ax1.axhline(y=0, color='red', linestyle='--')
    # ax1.spines['bottom'].set_color('grey')
    # ax1.xaxis.label.set_color('grey')
    # ax1.tick_params(axis='x', colors='grey')
    # ax1.spines['left'].set_color('grey')
    # ax1.yaxis.label.set_color('grey')
    # ax1.tick_params(axis='y', colors='grey')
    # ax1.set_yticks([-10000, 0, 50000])
    plt.title("Profit per case flagged")
    plt.xlabel("Model Threshold")
    plt.ylabel("US Dollars")
    plt.xlim([.05, 1])
    plt.ylim([-5, max_profit + 10])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)
    else:
        plt.show()
    return

if __name__ == '__main__':
    plot_avg_profits()
