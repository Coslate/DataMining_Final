#! /usr/bin/env python3.6
'''
    Author      : Coslate
    Date        : 2018/05/16
    Description :
        This program demonstrate the example to use the method defined by
        "CrossValidationGridSearchNested" to do Cross Validation Grid Search
        of a model.
        The purpose is to estimate the best parameter through two levels
        cross validation.
'''

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import numpy as np
import operator

#########################
#     Main-Routine      #
#########################
def main():
    data = load_breast_cancer()
    X = data.data
    Y = data.target

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                        'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    # Number of random trials
    NUM_TRIALS = 30

    # We will use a Support Vector Classifier with "rbf" kernel
    svm = SVC()

    # Using the function to get best estimator
    (max_score, svm_best_estimator) = CrossValidationGridSearchNested(X, Y, NUM_TRIALS, 10, svm, tuned_parameters, 'roc_auc')
    svm_best_parameter = svm_best_estimator.get_params()

    print(f'\nmax_score = {max_score}\n')
    print(f'\nbest_estimator = {svm_best_estimator}\n')
    print(f'\nbest_parameter = {svm_best_parameter}\n')

#########################
#     Sub-Routine       #
#########################
def CrossValidationGridSearchNested(X_data, Y_data, num_trials, fold_num, est_classifcation, tuned_param, scoring):
    max_score = -1
    best_estimator = est_classifcation

    for i in range(num_trials):
        inner_cv = StratifiedKFold(n_splits=fold_num, random_state=i, shuffle=True)
        outer_cv = StratifiedKFold(n_splits=fold_num, random_state=i+1, shuffle=True)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=est_classifcation, param_grid=tuned_param, cv=inner_cv, scoring=scoring)
        clf.fit(X_data, Y_data)

        # CV with parameter optimization
        param_score = cross_val_score(clf.best_estimator_, X=X_data, y=Y_data, cv=outer_cv, scoring=scoring).mean()
        if(param_score > max_score):
            max_score = param_score
            best_estimator = clf.best_estimator_

        progress = i/num_trials*100
        print(f'> progress = {progress}%')
    return (max_score, best_estimator)

#---------------Execution---------------#
if __name__ == '__main__':
    main()
