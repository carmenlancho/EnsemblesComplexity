import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from All_measures import all_measures
import random # for sampling with weights
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from aux_functions import aggregation_results_final_algorithm_cycle
import math
import multiprocessing as mp

root_path = os.getcwd()

# path_csv = os.chdir(root_path+'/datasets')
# # Extraemos los nombres de todos los ficheros
# total_name_list = []
# for filename in os.listdir(path_csv):
#     if filename.endswith('.csv'):
#         total_name_list.append(filename)
#
# # yeast da problemas porque una clase es muy pequeña y no aparece en todos los folds (creo que tb es por DCP)
# # haberman da problemas y es por DCP que da solo dos valores y concuerdan con la y
#
# total_name_list = ['bands.csv']
#
#
# for data_file in total_name_list:
#     os.chdir(root_path + '/datasets')
#     print(data_file)
#     file = data_file
#     name_data = data_file[:-4]
#     data = pd.read_csv(file)
#
# method_weights = 'classic'
# # Get X (features) and y (target)
# X = data.iloc[:,:-1].to_numpy() # all variables except y
# X = preprocessing.scale(X)
# y = data[['y']].to_numpy().reshape(-1)
# y[y==0] = -1 # sign format
#
# M = 20 # number of models, ensemble size

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
# method_weights = 'error_w' #'classic' #'error_w'
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
    elif (method_weights == 'init_hard'):
        # comienzo con mayor peso a los puntos difíciles
        # Get complexity measure on train set
        data_train = pd.DataFrame(X_train)
        y_cm = y_train.copy()
        y_cm[y_cm == -1] = 0  # not sign format
        data_train['y'] = y_cm
        data_train.columns = data.columns
        df_measures, _ = all_measures(data_train,False,None, None)
        CM_values = df_measures[CM_selected]
        ranking_hard = CM_values.rank(method='average', ascending=True)  # more weight to difficult
        weights_v = ranking_hard / sum(ranking_hard)  # probability distribution
        weights_v = np.array(weights_v)
    elif (method_weights == 'init_easy_x2'):
        # comienzo con mayor peso a los puntos fáciles
        # Get complexity measure on train set
        data_train = pd.DataFrame(X_train)
        y_cm = y_train.copy()
        y_cm[y_cm == -1] = 0  # not sign format
        data_train['y'] = y_cm
        data_train.columns = data.columns
        df_measures, _ = all_measures(data_train, False, None, None)
        CM_values = df_measures[CM_selected]
        ranking_easy = CM_values.rank(method='average', ascending=False)  # more weight to easy
        # Even more weight to easy
        factor = 1.5
        ranking_easy_w = ranking_easy**factor
        weights_v = ranking_easy_w / sum(ranking_easy_w)  # probability distribution
        weights_v = np.array(weights_v)
    elif (method_weights == 'init_hard_x2'):
        # comienzo con mayor peso a los puntos difíciles
        # Get complexity measure on train set
        data_train = pd.DataFrame(X_train)
        y_cm = y_train.copy()
        y_cm[y_cm == -1] = 0  # not sign format
        data_train['y'] = y_cm
        data_train.columns = data.columns
        df_measures, _ = all_measures(data_train,False,None, None)
        CM_values = df_measures[CM_selected]
        ranking_hard = CM_values.rank(method='average', ascending=True)  # more weight to difficult
        # Even more weight to difficult
        factor = 1.5
        ranking_hard_w = ranking_hard ** factor
        weights_v = ranking_hard_w / sum(ranking_hard_w)  # probability distribution
        weights_v = np.array(weights_v)
    elif (method_weights == 'error_w_hard'):
        weights_v = np.ones(n_train) / n_train # por ahora empezamos con los pesos iniciales normales
        # Get complexity measure on train set
        data_train = pd.DataFrame(X_train)
        y_cm = y_train.copy()
        y_cm[y_cm == -1] = 0  # not sign format
        data_train['y'] = y_cm
        data_train.columns = data.columns
        df_measures, _ = all_measures(data_train, False, None, None)
        CM_values = df_measures[CM_selected] # more weight to difficult
        factor_lambda = 0.025
    elif (method_weights == 'error_w_easy'):
        weights_v = np.ones(n_train) / n_train # por ahora empezamos con los pesos iniciales normales
        # Get complexity measure on train set
        data_train = pd.DataFrame(X_train)
        y_cm = y_train.copy()
        y_cm[y_cm == -1] = 0  # not sign format
        data_train['y'] = y_cm
        data_train.columns = data.columns
        df_measures, _ = all_measures(data_train, False, None, None)
        CM_values = 1 - df_measures[CM_selected] # more weight to easy
        factor_lambda = 0.05

    for m in range(M):
        # print(m)
        # m = 0

        # Fit a classifier with the specific weights
        random.seed(1)
        clf_m = DecisionTreeClassifier(random_state=0, max_depth=1)
        clf_m.fit(X_train, y_train, sample_weight=weights_v)
        # We append the classifier to the list
        clf_list.append(clf_m)

        # Prediction and calculation of error
        y_pred = clf_m.predict(X_train)
        preds_train.append(y_pred)
        disagree = np.not_equal(y_train, y_pred)
        if (method_weights == 'error_w_easy') | (method_weights == 'error_w_hard'):
            error_m = (sum(weights_v * (1 + factor_lambda*CM_values) * disagree)) / sum(weights_v * (1 + factor_lambda*CM_values))
        else:
            error_m = (sum(weights_v * disagree)) / sum(weights_v)


        # Compute alpha_m
        if (error_m != 0.0):
            alpha_m = 1/2 * np.log((1 - error_m) / error_m)
        else:
            alpha_m = 0.00001 # since there is no error, there is no change in weights
            # mejor hacer error_m = 0.00001 en pla muy bajito y dejar que tire
        alpha_list.append(alpha_m)

        # Evaluate on test
        y_pred_test = clf_m.predict(X_test)
        preds_test.append(y_pred_test)

        # Update the observations weights
        weights_v = weights_v * np.exp(-alpha_m * y_train * y_pred)

        # Weight normalization
        weights_v = weights_v / np.sum(weights_v)




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
# M = 200  # number of models, ensemble size
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
# CM_selected = 'Hostility'
# n_cv_splits = 10
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
        # print(fold)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        final_pred_train, final_pred_test, exp_loss_avg, misc_rate, misc_rate_test,conf_matrix = boosting_algorithm(X_train,y_train,X_test,y_test,M,
                                                                                                                    method_weights,CM_selected, plot_error)

        fold_v = [fold]*M
        results_dict = {'dataset':dataset_v,'fold':fold_v,'n_ensemble':n_ensemble_v,'method_weights':weights_type,
                        'compl_measure':CM_selected_v,
                        'exp_loss_avg_train':exp_loss_avg,'misc_rate_train':misc_rate,
                        'misc_rate_test':misc_rate_test,'conf_matrix_test':conf_matrix}

        results_aux = pd.DataFrame(results_dict)
        results = pd.concat([results, results_aux])
        results.reset_index(drop=True, inplace=True)

    # Aggregation per fold
    res_agg = aggregation_results_boosting(results)

    return results, res_agg


# dataset = 'bands'
# n_cv_splits = 10
# plot_error = True
# method_weights = 'init_easy'
# # method_weights = 'classic'
# # method_weights = 'init_easy_x2'
# CM_selected = 'kDN'
# M=200
# results, res_agg = CV_boosting(dataset,X,y,M,method_weights,CM_selected, plot_error,n_cv_splits)


# Función para sacarlo para todas las medidas de complejidad

def boosting_all_combinations(path_to_save, dataset, X,y):
    CM_list = ['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD', 'N1', 'N2', 'LSC', 'F1']
    # method_weights_list = ['classic','init_easy','init_hard','init_easy_x2','init_hard_x2',
    #                        'error_w_easy','error_w_hard']
    method_weights_list = ['classic','error_w_easy','error_w_hard']

    # Para guardar todos los resultados
    results_total = pd.DataFrame(columns=['dataset','fold','n_ensemble','method_weights','compl_measure',
                                          'exp_loss_avg_train','misc_rate_train',
                                          'misc_rate_test','conf_matrix_test'])
    res_agg_total = pd.DataFrame(columns=['dataset', 'n_ensemble', 'method_weights', 'compl_measure',
                                            'exp_loss_avg_train_mean', 'misc_rate_train_mean',
                                             'misc_rate_test_mean', 'exp_loss_avg_train_std', 'misc_rate_train_std',
                                             'misc_rate_test_std', 'conf_matrix_test_total'])

    # Algunos parámetros que dejamos fijos
    M = 300 # 20
    n_cv_splits = 10 # 5
    plot_error = False


    ## Caso clásico
    method_weights = 'classic'
    CM_selected = 'none'
    results, res_agg = CV_boosting(dataset, X, y, M, method_weights, CM_selected, plot_error, n_cv_splits)
    results_total = pd.concat([results_total, results])
    res_agg_total = pd.concat([res_agg_total, res_agg])

    # Including complexity
    method_weights_list.remove('classic')
    for method_weights in method_weights_list:
        print(method_weights)
        for CM_selected in CM_list:
            print(CM_selected)
            results, res_agg = CV_boosting(dataset, X, y, M, method_weights, CM_selected, plot_error, n_cv_splits)

            results_total = pd.concat([results_total, results])
            res_agg_total = pd.concat([res_agg_total, res_agg])



    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'Results_Boosting_' + dataset + '_factor005.csv'
    nombre_csv_aggr = 'AggregatedResults_Boosting_' + dataset + '_factor005.csv'
    results_total.to_csv(nombre_csv, encoding='utf_8_sig',index=False)
    res_agg_total.to_csv(nombre_csv_aggr, encoding='utf_8_sig', index=False)

    return results_total, res_agg_total




path_csv = os.chdir(root_path+'/datasets')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv'):
        total_name_list.append(filename)


 #'segment.csv', # da error porque es multiclase, no lo usamos

#total_name_list = ['bands.csv']
total_name_list = ['teaching_assistant_MH.csv','contraceptive_NL.csv','hill_valley_without_noise_traintest.csv',
 'glass0.csv','saheart.csv','breast-w.csv','contraceptive_LS.csv', 'yeast1.csv','ilpd.csv',
    'phoneme.csv','mammographic.csv','contraceptive_NS.csv','bupa.csv','Yeast_CYTvsNUC.csv','ring.csv','titanic.csv',
 'musk1.csv','spectfheart.csv','arrhythmia_cfs.csv','vertebral_column.csv','profb.csv','sonar.csv',
 'liver-disorders.csv','steel-plates-fault.csv','credit-g.csv','glass1.csv',
 'breastcancer.csv', 'diabetes.csv',
    'diabetic_retinopathy.csv', 'WineQualityRed_5vs6.csv',
 'teaching_assistant_LM.csv', 'ionosphere.csv', 'bands.csv',
 'wdbc.csv',
 'sylvine.csv',
 'teaching_assistant_LH.csv',
 'vehicle2.csv',
 'pima.csv',
 'spambase.csv',
 'fri_c0_250_50.csv',
 'parkinsons.csv',
'bodyfat.csv',
 'banknote_authentication.csv',
 'chatfield_4.csv'
]

path_to_save = root_path + '/Results_Boosting_weights'
for data_file in total_name_list:
    os.chdir(root_path + '/datasets')
    print(data_file)
    file = data_file
    #file = 'breastcancer.csv'
    name_data = data_file[:-4]
    #dataset = name_data
    data = pd.read_csv(file)
    # Get X (features) and y (target)
    X = data.iloc[:, :-1].to_numpy()  # all variables except y
    X = preprocessing.scale(X)
    y = data[['y']].to_numpy().reshape(-1)
    y[y == 0] = -1  # sign format

    _, _ = boosting_all_combinations(path_to_save, name_data, X, y)



# n_cv_splits = 5
# skf = StratifiedKFold(n_splits=n_cv_splits, random_state=1, shuffle=True)
# fold = 0
# for train_index, test_index in skf.split(X, y):
#     fold = fold + 1
#     # print(fold)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
# M=20
# plot_error = True
# def logitboost_algorithm(X_train, y_train, X_test, y_test, M, method_weights, CM_selected, plot_error):
#     # Asegurarse de que y_train e y_test tengan valores en {-1, 1}
#     if any(y_train == 0):
#         y_train[y_train == 0] = -1
#     if any(y_test == 0):
#         y_test[y_test == 0] = -1
#
#     n_train = len(y_train)
#     n_test = len(y_test)
#
#     # Listas para almacenar los modelos y alphas
#     clf_list = []
#
#     # Variables para guardar las predicciones
#     preds_train = np.zeros(n_train)
#     preds_test = np.zeros(n_test)
#
#     # Almacenar las pérdidas y las tasas de error
#     log_loss_avg = []
#     misc_rate = []
#     misc_rate_test = []
#     conf_matrix = []
#
#     # Inicialización de pesos
#     weights_v = np.ones(n_train) / n_train
#
#     for m in range(M):
#         print(m)
#         # Calcular las probabilidades actuales
#         p_train = 1 / (1 + np.exp(-2*preds_train))
#
#         # Calcular las pseudorespuestas y los pesos
#         z_i = (y_train - p_train) / (p_train * (1 - p_train))
#         w_i = p_train * (1 - p_train)
#
#         # Ajustar el modelo débil en las pseudorespuestas con pesos
#         clf_m = DecisionTreeRegressor(max_depth=1, random_state=0)
#         clf_m.fit(X_train, z_i, sample_weight=w_i)
#         clf_list.append(clf_m)
#
#         # Predicciones del modelo débil y actualización
#         y_pred_m = clf_m.predict(X_train)
#         preds_train += 0.5 * y_pred_m
#         preds_test += 0.5 * clf_m.predict(X_test)
#
#         # Calcular la pérdida logarítmica promedio en el conjunto de entrenamiento
#         log_loss_m = -np.mean(y_train * preds_train - np.log(1 + np.exp(preds_train)))
#         log_loss_avg.append(log_loss_m)
#
#         # Calcular la tasa de error en entrenamiento y prueba
#         misc_rate_m = np.mean(np.sign(preds_train) != y_train)
#         misc_rate.append(misc_rate_m)
#
#         misc_rate_test_m = np.mean(np.sign(preds_test) != y_test)
#         misc_rate_test.append(misc_rate_test_m)
#
#         # Matriz de confusión
#         conf_matrix.append(confusion_matrix(y_test, np.sign(preds_test)).tolist())
#
#     # Predicciones finales en entrenamiento y prueba
#     final_pred_train = np.sign(preds_train)
#     final_pred_test = np.sign(preds_test)
#
#     # Plot de la pérdida logarítmica y tasa de error si es necesario
#     if plot_error:
#         iterations = np.arange(1, M + 1)
#         plt.plot(iterations, log_loss_avg, label="Average log-loss", color='#1AB7D3')
#         plt.plot(iterations, misc_rate, label="Misclassification rate", color='crimson')
#         plt.xlabel('Number of boosting iterations')
#         plt.legend()
#         plt.show()
#
#     return final_pred_train, final_pred_test, log_loss_avg, misc_rate, misc_rate_test, conf_matrix
#




















