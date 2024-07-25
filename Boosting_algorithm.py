import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from All_measures import all_measures
import random # for sampling with weights
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from aux_functions import aggregation_results_final_algorithm_cycle
import math
import multiprocessing as mp

root_path = os.getcwd()

path_csv = os.chdir(root_path+'/datasets')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv'):
        total_name_list.append(filename)

# yeast da problemas porque una clase es muy pequeña y no aparece en todos los folds (creo que tb es por DCP)
# haberman da problemas y es por DCP que da solo dos valores y concuerdan con la y

total_name_list = ['bands.csv']


for data_file in total_name_list:
    os.chdir(root_path + '/datasets')
    print(data_file)
    file = data_file
    name_data = data_file[:-4]
    data = pd.read_csv(file)

method_weights = 'classic'
# Get X (features) and y (target)
X = data.iloc[:,:-1].to_numpy() # all variables except y
X = preprocessing.scale(X)
y = data[['y']].to_numpy().reshape(-1)
y[y==0] = -1 # sign format

M = 2000 # number of models, ensemble size


skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
fold = 0
for train_index, test_index in skf.split(X, y):
    fold = fold + 1
    print(fold)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    n = len(y_train)
    n_test = len(y_test)

    # List to save classifiers
    clf_list = []
    # List to save alphas
    alpha_list = []

    # Preds_train
    preds_train = []
    # Preds_test
    preds_test = []
    # Exponential loss
    exp_loss_avg = []
    # Misclassification rate
    misc_rate = []


    # Initialize weights
    if (method_weights == 'classic'):
        weights_v = np.ones(n) / n
    else: # algo de complejidad
        weights_v = np.ones(n) / n # sera una función de la complejidad


    for m in range(M):
        print(m)
        # m = 0

        # Fit a classifier with the specific weights
        clf_m = DecisionTreeClassifier(random_state=0,max_depth=1)
        clf_m.fit(X_train, y_train, sample_weight=weights_v)
        # We append the classifier to the list
        clf_list.append(clf_m)


        # Prediction and calculation of error
        y_pred = clf_m.predict(X_train)
        preds_train.append(y_pred)
        disagree = np.not_equal(y_train, y_pred)
        error_m = (sum(weights_v * disagree)) / sum(weights_v)


        # Compute alpha_m
        alpha_m = np.log((1-error_m)/error_m)
        alpha_list.append(alpha_m)

        # Evaluate on test
        y_pred_test = clf_m.predict(X_test)
        preds_test.append(y_pred_test)

        # Update the observations weights
        weights_v[disagree] = weights_v[disagree] * np.exp(alpha_m)


    # Predictions for train and test
    df_preds_train = np.zeros(n)
    df_preds_test = np.zeros(n_test)
    for m in range(M):
        df_preds_train = df_preds_train + (preds_train[m] * alpha_list[m])
        exp_loss_avg_m = (1/n)*np.sum(np.exp(-np.sign(df_preds_train)*y_train))
        exp_loss_avg.append(exp_loss_avg_m)
        misc_rate_m = np.sum(np.not_equal(y_train, np.sign(df_preds_train)))/n
        misc_rate.append(misc_rate_m)
        df_preds_test = df_preds_test + (preds_test[m] * alpha_list[m])

    final_pred_train = np.sign(df_preds_train)
    final_pred_test = np.sign(df_preds_test)

    # Plot misclassification rate and exponential loss
    iterations = np.arange(1,M+1)
    plt.plot(iterations, exp_loss_avg, label="Average exponential loss",color='#1AB7D3')
    plt.plot(iterations, misc_rate, label="Misclassification rate", color='crimson')
    plt.xlabel('Number of boosting iterations')
    plt.legend()
    plt.show()






