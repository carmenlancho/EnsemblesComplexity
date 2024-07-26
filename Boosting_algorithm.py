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

M = 20 # number of models, ensemble size

#
# skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
# fold = 0
# for train_index, test_index in skf.split(X, y):
#     fold = fold + 1
#     print(fold)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     n = len(y_train)
#     n_test = len(y_test)
#
#     # List to save classifiers
#     clf_list = []
#     # List to save alphas
#     alpha_list = []
#
#     # Preds_train
#     preds_train = []
#     # Preds_test
#     preds_test = []
#     # Exponential loss
#     exp_loss_avg = []
#     # Misclassification rate
#     misc_rate = []
#
#
#     # Initialize weights
#     if (method_weights == 'classic'):
#         weights_v = np.ones(n) / n
#     else: # algo de complejidad
#         weights_v = np.ones(n) / n # sera una función de la complejidad
#
#
#     for m in range(M):
#         print(m)
#         # m = 0
#
#         # Fit a classifier with the specific weights
#         clf_m = DecisionTreeClassifier(random_state=0,max_depth=1)
#         clf_m.fit(X_train, y_train, sample_weight=weights_v)
#         # We append the classifier to the list
#         clf_list.append(clf_m)
#
#
#         # Prediction and calculation of error
#         y_pred = clf_m.predict(X_train)
#         preds_train.append(y_pred)
#         disagree = np.not_equal(y_train, y_pred)
#         error_m = (sum(weights_v * disagree)) / sum(weights_v)
#
#
#         # Compute alpha_m
#         alpha_m = np.log((1-error_m)/error_m)
#         alpha_list.append(alpha_m)
#
#         # Evaluate on test
#         y_pred_test = clf_m.predict(X_test)
#         preds_test.append(y_pred_test)
#
#         # Update the observations weights
#         weights_v[disagree] = weights_v[disagree] * np.exp(alpha_m)
#
#
#     # Predictions for train and test
#     df_preds_train = np.zeros(n)
#     df_preds_test = np.zeros(n_test)
#     for m in range(M):
#         df_preds_train = df_preds_train + (preds_train[m] * alpha_list[m])
#         exp_loss_avg_m = (1/n)*np.sum(np.exp(-np.sign(df_preds_train)*y_train))
#         exp_loss_avg.append(exp_loss_avg_m)
#         misc_rate_m = np.sum(np.not_equal(y_train, np.sign(df_preds_train)))/n
#         misc_rate.append(misc_rate_m)
#         df_preds_test = df_preds_test + (preds_test[m] * alpha_list[m])
#
#     final_pred_train = np.sign(df_preds_train)
#     final_pred_test = np.sign(df_preds_test)
#
#     # Plot misclassification rate and exponential loss
#     iterations = np.arange(1,M+1)
#     plt.plot(iterations, exp_loss_avg, label="Average exponential loss",color='#1AB7D3')
#     plt.plot(iterations, misc_rate, label="Misclassification rate", color='crimson')
#     plt.xlabel('Number of boosting iterations')
#     plt.legend()
#     plt.show()


## Función general
# method_weights = 'classic'
# plot_error = False
# M = 20  # number of models, ensemble size
# CM_selected = 'Hostility'
def boosting_algorithm(X_train,y_train,X_test,y_test,M,method_weights,CM_selected, plot_error):
    # X_train and X_test are already preprocessed
    # y in {-1,1}
    if any(y_train==0):
        y_train[y_train == 0] = -1  # sign format
    if any(y_test==0):
        y_test[y_test == 0] = -1  # sign format

    n_train = len(y_train)
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
    misc_rate_test = []
    # confusion matrix
    conf_matrix = []

    # Initialize weights
    if (method_weights == 'classic'):
        weights_v = np.ones(n_train) / n_train
    elif (method_weights == 'init_easy'):
        # comienzo con mayor peso a los puntos fáciles
        # Get complexity measure on train set
        data_train = pd.DataFrame(X_train)
        y_cm = y_train.copy()
        y_cm[y_cm == -1] = 0  # not sign format
        data_train['y'] = y_cm
        data_train.columns = data.columns
        df_measures, _ = all_measures(data_train,False,None, None)
        CM_values = df_measures[CM_selected]
        ranking_easy = CM_values.rank(method='average', ascending=False)  # more weight to easy
        weights_v = ranking_easy / sum(ranking_easy)  # probability distribution
        weights_v = np.array(weights_v)
    elif (method_weights == 'init_easy_w_complex'):
        # Get complexity measure on train set
        data_train = pd.DataFrame(X_train)
        y_cm = y_train.copy()
        y_cm[y_cm == -1] = 0  # not sign format
        data_train['y'] = y_cm
        data_train.columns = data.columns
        df_measures, _ = all_measures(data_train,False,None, None)
        CM_values = df_measures[CM_selected]
        # Para el inicio
        ranking_easy = CM_values.rank(method='average', ascending=False)  # more weight to easy
        weights_v = ranking_easy / sum(ranking_easy)  # probability distribution
        weights_v = np.array(weights_v)
        # Para el update de los pesos
        ranking_hard = CM_values.rank(method='average', ascending=True)  # more weight to difficult
        weights_hard = ranking_hard / sum(ranking_hard)  # probability distribution
        weights_hard_v = np.array(weights_hard)

    for m in range(M):
        print(m)
        # m = 0

        # Fit a classifier with the specific weights
        clf_m = DecisionTreeClassifier(random_state=0, max_depth=1)
        clf_m.fit(X_train, y_train, sample_weight=weights_v)
        # We append the classifier to the list
        clf_list.append(clf_m)

        # Prediction and calculation of error
        y_pred = clf_m.predict(X_train)
        preds_train.append(y_pred)
        disagree = np.not_equal(y_train, y_pred)
        error_m = (sum(weights_v * disagree)) / sum(weights_v)

        # Compute alpha_m
        alpha_m = np.log((1 - error_m) / error_m)
        alpha_list.append(alpha_m)

        # Evaluate on test
        y_pred_test = clf_m.predict(X_test)
        preds_test.append(y_pred_test)

        # Update the observations weights
        if (method_weights=='classic') | (method_weights=='init_easy'):
            weights_v[disagree] = weights_v[disagree] * np.exp(alpha_m)
        else: # actualizamos tb en función de la complejidad
            update_boosting = weights_v[disagree] * np.exp(alpha_m)
            update_complexity = weights_hard_v[disagree]
            total_update_weights = (19/20)*update_boosting + (1/20)*update_complexity
            total_update_weights = total_update_weights/ sum(total_update_weights)
            weights_v[disagree] = total_update_weights




    # Predictions for train and test
    df_preds_train = np.zeros(n_train)
    df_preds_test = np.zeros(n_test)
    for m in range(M):
        df_preds_train = df_preds_train + (preds_train[m] * alpha_list[m])
        exp_loss_avg_m = (1 / n_train) * np.sum(np.exp(-np.sign(df_preds_train) * y_train))
        exp_loss_avg.append(exp_loss_avg_m)
        misc_rate_m = np.sum(np.not_equal(y_train, np.sign(df_preds_train))) / n_train
        misc_rate.append(misc_rate_m)
        df_preds_test = df_preds_test + (preds_test[m] * alpha_list[m])
        misc_rate_test_m = np.sum(np.not_equal(y_test, np.sign(df_preds_test))) / n_test
        misc_rate_test.append(misc_rate_test_m)
        final_pred_test_m = np.sign(df_preds_test)
        conf_matrix.append(confusion_matrix(y_test, final_pred_test_m).tolist())


    final_pred_train = np.sign(df_preds_train)
    final_pred_test = np.sign(df_preds_test)

    if plot_error:
        # Plot misclassification rate and exponential loss
        iterations = np.arange(1, M + 1)
        plt.plot(iterations, exp_loss_avg, label="Average exponential loss", color='#1AB7D3')
        plt.plot(iterations, misc_rate, label="Misclassification rate", color='crimson')
        plt.xlabel('Number of boosting iterations')
        plt.legend()
        plt.show()

    return final_pred_train, final_pred_test, exp_loss_avg, misc_rate, misc_rate_test, conf_matrix

# method_weights = 'classic'
# plot_error = False
# M = 20  # number of models, ensemble size
# final_pred_train, final_pred_test, exp_loss_avg, misc_rate, misc_rate_test, conf_matrix =  boosting_algorithm(X_train,y_train,X_test,y_test,M,method_weights, plot_error)


def aggregation_results_boosting(results):
    res_agg_mean = results.groupby(['dataset','n_ensemble','method_weights','compl_measure'], as_index=False)[['exp_loss_avg_train',
                                                                    'misc_rate_train','misc_rate_test']].mean()
    res_agg_mean.rename({'exp_loss_avg_train': 'exp_loss_avg_train_mean', 'misc_rate_train': 'misc_rate_train_mean',
                         'misc_rate_test':'misc_rate_test_mean'}, axis=1, inplace=True)

    res_agg_std = results.groupby(['dataset','n_ensemble','method_weights','compl_measure'], as_index=False)[['exp_loss_avg_train',
                                                                    'misc_rate_train','misc_rate_test']].std()
    res_agg_std.rename({'exp_loss_avg_train': 'exp_loss_avg_train_std', 'misc_rate_train': 'misc_rate_train_std',
                         'misc_rate_test':'misc_rate_test_std'}, axis=1, inplace=True)

    res_agg_confmatrix = results.groupby(['dataset','n_ensemble', 'method_weights','compl_measure'])['conf_matrix_test'].apply(lambda x: np.sum(np.array(x.tolist()), axis=0).tolist())
    res_agg_confmatrix = pd.DataFrame(res_agg_confmatrix)
    res_agg_confmatrix.reset_index(inplace=True)
    res_agg_confmatrix.rename({'conf_matrix_test': 'conf_matrix_test_total'}, axis=1, inplace=True)
    # All together in a dataframe
    res_agg = pd.merge(res_agg_mean, res_agg_std[['n_ensemble', 'exp_loss_avg_train_std',
       'misc_rate_train_std', 'misc_rate_test_std']], left_on=['n_ensemble'], right_on=['n_ensemble'])

    res_agg = pd.merge(res_agg, res_agg_confmatrix[['n_ensemble', 'conf_matrix_test_total']],
                       left_on=['n_ensemble'], right_on=['n_ensemble'])

    return res_agg



### Cross-Validation Boosting
# dataset='aa'
# M = 20
def CV_boosting(dataset,X,y,M,method_weights,CM_selected, plot_error,n_cv_splits):

    if any(y==0):
        y[y == 0] = -1  # sign format
    if method_weights == 'classic':
        CM_selected = 'none'

    dataset_v = [dataset]*M
    n_ensemble_v = list(np.arange(1,M+1))
    weights_type = [method_weights] * M
    CM_selected_v = [CM_selected]*M

    # dataframe to save the results
    results = pd.DataFrame(columns=['dataset','fold','n_ensemble','method_weights','compl_measure','exp_loss_avg_train',
                                    'misc_rate_train','misc_rate_test','conf_matrix_test'])


    skf = StratifiedKFold(n_splits=n_cv_splits, random_state=1, shuffle=True)
    fold = 0
    for train_index, test_index in skf.split(X, y):
        fold = fold + 1
        print(fold)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        final_pred_train, final_pred_test, exp_loss_avg, misc_rate, misc_rate_test,conf_matrix = boosting_algorithm(X_train,y_train,X_test,y_test,M,
                                                                                                                    method_weights,CM_selected, plot_error)

        fold_v = [fold]*M
        results_dict = {'dataset':dataset_v,'fold':fold_v,'n_ensemble':n_ensemble_v,'method_weights':weights_type,
                        'compl_measure':CM_selected_v,
                        'exp_loss_avg_train':exp_loss_avg,'misc_rate_train':misc_rate,
                        'misc_rate_test':misc_rate_test,'conf_matrix_test':conf_matrix} # falta por añadir confusion matrix

        results_aux = pd.DataFrame(results_dict)
        results = pd.concat([results, results_aux])
        results.reset_index(drop=True, inplace=True)

    # Aggregation per fold
    res_agg = aggregation_results_boosting(results)

    return results, res_agg


dataset = 'bands'
n_cv_splits = 10
plot_error = True
method_weights = 'init_easy'
# method_weights = 'classic'
# method_weights = 'init_easy_w_complex'
CM_selected = 'kDN'
M=200
results, res_agg = CV_boosting(dataset,X,y,M,method_weights,CM_selected, plot_error,n_cv_splits)













