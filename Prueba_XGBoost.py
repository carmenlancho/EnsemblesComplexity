# Si queremos hacerlo con xgboost, podemos usar una función de pérdida customizada
# pero también necesitamos la hessiana


import numpy as np
import xgboost as xgb

# Definimos nuestra propia función de pérdida
def complexity_weighted_logloss(complexity_array):
    def custom_logloss(preds, dtrain):
        y_true = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))  # Convert logits to probabilities

        # Compute weighted gradients and Hessians
        grad = complexity_array * (preds - y_true)
        hess = complexity_array * preds * (1 - preds)

        return grad, hess
    return custom_logloss

# Train XGBoost model with custom loss



dtrain = xgb.DMatrix(X_train, label=y_train, weight=complexity_array)

params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Train using your custom objective
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    obj=complexity_weighted_logloss(complexity_array)
)

# to evaluate your model with the same complexity-aware loss, you can define a custom evaluation function

def complexity_weighted_logloss_eval(complexity_array):
    def eval_metric(preds, dtrain):
        y_true = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))

        eps = 1e-15  # To avoid log(0)
        logloss = -np.sum(
            complexity_array * (
                y_true * np.log(preds + eps) +
                (1 - y_true) * np.log(1 - preds + eps)
            )
        ) / np.sum(complexity_array)

        return 'complexity_logloss', logloss
    return eval_metric


bst = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    obj=complexity_weighted_logloss(complexity_array),
    feval=complexity_weighted_logloss_eval(complexity_array),
    evals=[(dtrain, 'train')],
    verbose_eval=True
)



### Código para hacer pruebas de complejidad con Gradient Boosting
### Vamos a cambiar los pesos iniciales
### y los pesos del fit

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from All_measures import all_measures
#import random # for sampling with weights
from sklearn import preprocessing
#from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import log_loss


root_path = os.getcwd()

# Tengo que hacer:
# cambios con el init: fácil y difícil POR HACER
# cambios con el sample_weights: fácil y difícil
# combo de cambios
# hay 2 funciones pérdida log-loss y exponential (que es adaboost)

#####################################################################################################
### ESTO NO HE LOGRADO QUE FUNCIONE POR TEMAS DE TAMAÑO MUESTRAL DE LA PREDICCIÓN DE TEST

## Con esta función puedo modificar los pesos iniciales (MÁS BIEN LAS PREDICCIONES INICIALES)
#
# class CustomInitModel(BaseEstimator, ClassifierMixin):
#     def __init__(self, initial_weights):
#         # Inicializamos los pesos, que deben tener un tamaño igual al número de instancias
#         self.initial_weights = initial_weights
#
#     def fit(self, X, y):
#         # No necesitamos entrenar, ya que las predicciones son solo basadas en los pesos
#         return self
#
#     # def predict_proba(self, X):
#     #     # Devuelve las probabilidades como una matriz con dos columnas
#     #     proba_pos_class = np.array(self.initial_weights).reshape(-1, 1)  # Probabilidades para la clase positiva (1)
#     #     proba_neg_class = 1 - proba_pos_class  # Probabilidades para la clase negativa (0)
#     #     proba = np.hstack([proba_neg_class, proba_pos_class])  # Concatenamos ambas columnas
#     #     return proba
#
#     def predict_proba(self, X):
#         if len(X) != len(self.initial_weights):
#             raise ValueError("El tamaño de X no coincide con el de los pesos iniciales")
#         proba_pos_class = np.array(self.initial_weights[:len(X)]).reshape(-1, 1)  # Solo las instancias relevantes
#         proba_neg_class = 1 - proba_pos_class
#         proba = np.hstack([proba_neg_class, proba_pos_class])
#         return proba
#
#
# # Vector de pesos iniciales para dos clases
# initial_weights = np.array([0.7, 0.3])  # Clase 0: 70%, Clase 1: 30%
#
# # Usar el modelo inicial personalizado
# init_model = CustomInitModel(initial_weights=initial_weights)
#
#
# # Pesos iniciales para cada instancia
# initial_weights = [0.1, 0.5, 0.3, 0.7]
#
# # Datos de ejemplo X y etiquetas y
# X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
# y = np.array([0, 1, 0, 1])
#
# from sklearn.datasets import make_classification
# # Crear datos de ejemplo
# X, y = make_classification(n_samples=200, n_features=5, random_state=42)
#
# skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
# fold = 0
# for train_index, test_index in skf.split(X, y):
#     fold = fold + 1
#     # print(fold)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
# len(X_test)
# len(y_test)
# # Get complexity measure on train set
# data_train = pd.DataFrame(X_train)
# y_cm = y_train.copy()
# data_train['y'] = y_cm
# data_train.columns = ['x1','x2','x3','x4','x5','y']
# df_measures, _ = all_measures(data_train, False, None, None)
# CM_selected = 'Hostility'
# CM_values = df_measures[CM_selected]
# # Quiero dar más peso a lo difícil
# initial_weights_1 = np.zeros(len(X_train))
# initial_weights_1[y_cm == 0] = CM_values[y_cm == 0]
# initial_weights_1[y_cm == 1] = 1 - CM_values[y_cm == 1]
#
# # Más complejidad a lo fácil
# initial_weights_1 = np.zeros(len(X))
# initial_weights_1[y_cm == 0] = 1 - CM_values[y_cm == 0]
# initial_weights_1[y_cm == 1] = CM_values[y_cm == 1]
#
#
# # Inicializar el modelo con los pesos
# # initial_weights = np.ones(len(X))
# init_model = CustomInitModel(initial_weights=initial_weights_1)
#
# # Ajustar el modelo (en este caso no es necesario)
# init_model.fit(X_train, y_train)
#
# # Predecir las probabilidades (para cada instancia)
# proba = init_model.predict_proba(X_train)
# print(proba)
# proba2 = init_model.predict_proba(X_test)
# print(proba2)
#
#
#
# # Aplicarlo en GradientBoostingClassifier
# gbc = GradientBoostingClassifier(init=init_model, n_estimators=100, random_state=42)
# gbc.fit(X_train, y_train)
# gbc.predict_proba(X_train)
# gbc.predict(X_train)
# gbc.predict(X_test)
# len(y_test)
# len(X_test)
# len(test_pred)
# 160/40
#
# for stage_index, (train_pred, test_pred) in enumerate(zip(
#     gbc.staged_predict(X_train),
#     gbc.staged_predict(X_test),
# )):
#     print(f"Etapa {stage_index}:")
#     print(f"Tamaño de train_pred: {len(train_pred)}, Tamaño de y_train: {len(y_train)}")
#     print(f"Tamaño de test_pred: {len(test_pred)}, Tamaño de y_test: {len(y_test)}")
#

###############################################################################################################



## Función general
# method_weights = 'sample_weight_easy'
# plot_error = False
# M = 20  # number of models, ensemble size
# CM_selected = 'Hostility'
# loss_f = 'log_loss' # 'exponential'
def gradient_boosting_algorithm(X_train,y_train,X_test,y_test,M,method_weights,loss_f, CM_selected, plot_error):
    # X_train and X_test are already preprocessed
    # y in {0,1}
    if any(y_train==-1):
        y_train[y_train == -1] = 0
    if any(y_test==-1):
        y_test[y_test == -1] = 0

    # n_train = len(y_train)
    # n_test = len(y_test)

    # Loss per iteration
    #loss_iter = clf_m.train_score_ # esto es un resumen final, no por iteración
    loss_iter = []
    loss_iter_test = []

    # Listas para guardar accuracies por iteración
    train_accuracies = []
    test_accuracies = []
    # Listas para guardar las matrices de confusión
    confusion_matrices_train = []
    confusion_matrices_test = []


    # comenzamos con decision stump
    # va dentro de los ifelse
    # clf_m = GradientBoostingClassifier(n_estimators=M, learning_rate=0.1, random_state=28,
    #                                    loss = loss_f,max_depth=1)



    # Entrenar el modelo
    if (method_weights == 'classic'):
        clf_m = GradientBoostingClassifier(n_estimators=M, learning_rate=0.1, random_state=28,
                                           loss=loss_f, max_depth=1)
        clf_m.fit(X_train, y_train)
    elif (method_weights == 'sample_weight_easy'):
        clf_m = GradientBoostingClassifier(n_estimators=M, learning_rate=0.1, random_state=28,
                                           loss=loss_f, max_depth=1)
        # comienzo con mayor peso a los puntos fáciles
        # Get complexity measure on train set
        data_train = pd.DataFrame(X_train)
        y_cm = y_train.copy()
        data_train['y'] = y_cm
        data_train.columns = data.columns
        df_measures, _ = all_measures(data_train, False, None, None)
        CM_values = df_measures[CM_selected]
        ranking_easy = CM_values.rank(method='average', ascending=False)  # more weight to easy
        weights_v = ranking_easy / sum(ranking_easy)  # probability distribution
        weights_v = np.array(weights_v)
        clf_m.fit(X_train, y_train,sample_weight=weights_v)
    elif (method_weights == 'sample_weight_hard'):
        clf_m = GradientBoostingClassifier(n_estimators=M, learning_rate=0.1, random_state=28,
                                           loss=loss_f, max_depth=1)
        # comienzo con mayor peso a los puntos difíciles
        # Get complexity measure on train set
        data_train = pd.DataFrame(X_train)
        y_cm = y_train.copy()
        data_train['y'] = y_cm
        data_train.columns = data.columns
        df_measures, _ = all_measures(data_train, False, None, None)
        CM_values = df_measures[CM_selected]
        ranking_hard = CM_values.rank(method='average', ascending=True)  # more weight to difficult
        weights_v = ranking_hard / sum(ranking_hard)  # probability distribution
        weights_v = np.array(weights_v)
        clf_m.fit(X_train, y_train, sample_weight=weights_v)
    elif (method_weights == 'sample_weight_easy_x2'):
        clf_m = GradientBoostingClassifier(n_estimators=M, learning_rate=0.1, random_state=28,
                                           loss=loss_f, max_depth=1)
        # comienzo con mayor peso a los puntos fáciles
        # Get complexity measure on train set
        data_train = pd.DataFrame(X_train)
        y_cm = y_train.copy()
        data_train['y'] = y_cm
        data_train.columns = data.columns
        df_measures, _ = all_measures(data_train, False, None, None)
        CM_values = df_measures[CM_selected]
        ranking_easy = CM_values.rank(method='average', ascending=False)  # more weight to easy
        # Even more weight to easy
        factor = 1.5
        ranking_easy_w = ranking_easy ** factor
        weights_v = ranking_easy_w / sum(ranking_easy_w)  # probability distribution
        weights_v = np.array(weights_v)
        clf_m.fit(X_train, y_train, sample_weight=weights_v)
    elif (method_weights == 'sample_weight_hard_x2'):
        clf_m = GradientBoostingClassifier(n_estimators=M, learning_rate=0.1, random_state=28,
                                           loss=loss_f, max_depth=1)
        # comienzo con mayor peso a los puntos difíciles
        # Get complexity measure on train set
        data_train = pd.DataFrame(X_train)
        y_cm = y_train.copy()
        data_train['y'] = y_cm
        data_train.columns = data.columns
        df_measures, _ = all_measures(data_train, False, None, None)
        CM_values = df_measures[CM_selected]
        ranking_hard = CM_values.rank(method='average', ascending=True)  # more weight to difficult
        # Even more weight to difficult
        factor = 1.5
        ranking_hard_w = ranking_hard ** factor
        weights_v = ranking_hard_w / sum(ranking_hard_w)  # probability distribution
        weights_v = np.array(weights_v)
        clf_m.fit(X_train, y_train, sample_weight=weights_v)

    # elif (method_weights == 'init_easy'):
    #     # Get complexity measure on train set
    #     data_train = pd.DataFrame(X_train)
    #     y_cm = y_train.copy()
    #     data_train['y'] = y_cm
    #     data_train.columns = data.columns
    #     df_measures, _ = all_measures(data_train, False, None, None)
    #     CM_values = df_measures[CM_selected]
    #     # Más complejidad a lo fácil
    #     initial_weights_1 = np.zeros(len(X_train))
    #     initial_weights_1[y_cm == 0] = 1 - CM_values[y_cm == 0]
    #     initial_weights_1[y_cm == 1] = CM_values[y_cm == 1]
    #
    #     init_model = CustomInitModel(initial_weights=initial_weights_1)
    #     clf_m = GradientBoostingClassifier(n_estimators=M, learning_rate=0.1, random_state=28,
    #                                        loss = loss_f,max_depth=1,init=init_model)
    #     clf_m.fit(X_train, y_train)
    #
    # elif (method_weights == 'init_hard'):
    #     # Get complexity measure on train set
    #     data_train = pd.DataFrame(X_train)
    #     y_cm = y_train.copy()
    #     data_train['y'] = y_cm
    #     data_train.columns = data.columns
    #     df_measures, _ = all_measures(data_train, False, None, None)
    #     CM_values = df_measures[CM_selected]
    #     # Quiero dar más peso a lo difícil
    #     initial_weights_1 = np.zeros(len(X_train))
    #     initial_weights_1[y_cm == 0] = CM_values[y_cm == 0]
    #     initial_weights_1[y_cm == 1] = 1 - CM_values[y_cm == 1]
    #
    #     init_model = CustomInitModel(initial_weights=initial_weights_1)
    #     clf_m = GradientBoostingClassifier(n_estimators=M, learning_rate=0.1, random_state=28,
    #                                        loss = loss_f,max_depth=1,init=init_model)
    #     clf_m.fit(X_train, y_train)
    #

    # Evaluar el modelo en cada iteración
    for train_pred, test_pred in zip(clf_m.staged_predict(X_train), clf_m.staged_predict(X_test)):
        train_accuracies.append(accuracy_score(y_train, train_pred))
        test_accuracies.append(accuracy_score(y_test, test_pred))

        # Matriz de confusión para train y test
        cm_train = confusion_matrix(y_train, train_pred)
        cm_test = confusion_matrix(y_test, test_pred) # [[TN, FP], [FN, TP]]

        # Guardar las matrices
        # confusion_matrices_train.append(cm_train)
        # confusion_matrices_test.append(cm_test)
        confusion_matrices_train.append(cm_train.tolist())
        confusion_matrices_test.append(cm_test.tolist())

    # clf_m.predict_proba(X_train)

    # Calcular pérdidas por cada iteración en TRAIN
    for decision_values in clf_m.staged_decision_function(X_train):
        if (loss_f == 'log_loss'):
            # Calcular log-loss (probabilidades predichas)
            prob_pos = 1 / (1 + np.exp(-decision_values))  # Probabilidad para la clase positiva
            logloss_iter = log_loss(y_train, prob_pos)  # Usamos log_loss de sklearn
            loss_iter.append(logloss_iter)
        elif (loss_f == 'exponential'):
            # Calcular exponential loss
            exp_loss = np.mean(np.exp(-y_train * decision_values))  # Exponential loss
            loss_iter.append(exp_loss)

    # Calcular pérdidas por cada iteración en TEST
    for decision_values in clf_m.staged_decision_function(X_test):
        if (loss_f == 'log_loss'):
            # Calcular log-loss (probabilidades predichas)
            prob_pos = 1 / (1 + np.exp(-decision_values))  # Probabilidad para la clase positiva
            logloss_iter = log_loss(y_test, prob_pos)  # Usamos log_loss de sklearn
            loss_iter_test.append(logloss_iter)
        elif (loss_f == 'exponential'):
            # Calcular exponential loss
            exp_loss = np.mean(np.exp(-y_test * decision_values))  # Exponential loss
            loss_iter_test.append(exp_loss)

    if plot_error:
        # Plot misclassification rate and loss TRAIN
        misc_rate_train = np.ones(M) - train_accuracies
        iterations = np.arange(1, M + 1)
        plt.plot(iterations, loss_iter, label="Loss function in train", color='#1AB7D3')
        plt.plot(iterations, misc_rate_train, label="Misc rate train", color='crimson')
        plt.xlabel('Number of iterations')
        plt.legend()
        plt.show()

        # Plot misclassification rate and loss TEST
        misc_rate_test = np.ones(M) - test_accuracies
        iterations = np.arange(1, M + 1)
        plt.plot(iterations, loss_iter_test, label="Loss function in test", color='#1AB7D3')
        plt.plot(iterations, misc_rate_test, label="Misc rate test", color='crimson')
        plt.xlabel('Number of iterations')
        plt.legend()
        plt.show()

    return loss_iter, loss_iter_test, train_accuracies, test_accuracies, confusion_matrices_train, confusion_matrices_test

# method_weights = 'classic'
# plot_error = False
# M = 200  # number of models, ensemble size
# final_pred_train, final_pred_test, exp_loss_avg, misc_rate, misc_rate_test, conf_matrix =  boosting_algorithm(X_train,y_train,X_test,y_test,M,method_weights, plot_error)


def aggregation_results_gradientboosting(results):

    res_agg_mean = results.groupby(['dataset','n_ensemble','method_weights','compl_measure', 'loss_selected'], as_index=False)[['loss_train',
                                                                   'loss_test','train_acc','test_acc']].mean()
    res_agg_mean.rename({'loss_train': 'loss_train_mean', 'loss_test': 'loss_test_mean',
                         'train_acc': 'train_acc_mean',
                         'test_acc':'test_acc_mean'}, axis=1, inplace=True)

    res_agg_std = results.groupby(['dataset','n_ensemble','method_weights','compl_measure', 'loss_selected'], as_index=False)[['loss_train',
                                                                   'loss_test','train_acc','test_acc']].std()
    res_agg_std.rename({'loss_train': 'loss_train_std', 'loss_test': 'loss_test_std',
                        'train_acc': 'train_acc_std',
                         'test_acc':'test_acc_std'}, axis=1, inplace=True)

    res_agg_confmatrix1 = results.groupby(['dataset','n_ensemble','method_weights','compl_measure',
                                          'loss_selected'])['conf_matr_train'].apply(lambda x: np.sum(np.array(x.tolist()),axis=0).tolist())
    res_agg_confmatrix1 = pd.DataFrame(res_agg_confmatrix1)
    res_agg_confmatrix1.reset_index(inplace=True)
    res_agg_confmatrix1.rename({'conf_matr_train': 'conf_matr_train_total'}, axis=1, inplace=True)

    res_agg_confmatrix2 = results.groupby(['dataset','n_ensemble','method_weights','compl_measure',
                                          'loss_selected'])['conf_matr_test'].apply(lambda x: np.sum(np.array(x.tolist()),axis=0).tolist())
    res_agg_confmatrix2 = pd.DataFrame(res_agg_confmatrix2)
    res_agg_confmatrix2.reset_index(inplace=True)
    res_agg_confmatrix2.rename({'conf_matr_test': 'conf_matr_test_total'}, axis=1, inplace=True)

    # All together in a dataframe
    res_agg = pd.merge(res_agg_mean, res_agg_std[['n_ensemble', 'loss_train_std','loss_test_std',
       'train_acc_std', 'test_acc_std']], left_on=['n_ensemble'], right_on=['n_ensemble'])

    res_agg = pd.merge(res_agg, res_agg_confmatrix1[['n_ensemble', 'conf_matr_train_total']],
                       left_on=['n_ensemble'], right_on=['n_ensemble'])

    res_agg = pd.merge(res_agg, res_agg_confmatrix2[['n_ensemble','conf_matr_test_total']],
                       left_on=['n_ensemble'], right_on=['n_ensemble'])

    return res_agg



### Cross-Validation Boosting
# dataset='aa'
# M = 20
# CM_selected = 'Hostility'
# n_cv_splits = 10
# method_weights = 'init_easy'
# plot_error = False
# loss_f = 'log_loss'
def CV_Gradientboosting(dataset,X,y,M,method_weights,CM_selected, plot_error,n_cv_splits,loss_f):

    if any(y==-1):
        y[y == -1] = 0
    if method_weights == 'classic':
        CM_selected = 'none'

    dataset_v = [dataset]*M
    n_ensemble_v = list(np.arange(1,M+1))
    weights_type = [method_weights] * M
    CM_selected_v = [CM_selected]*M
    loss_f_v = [loss_f] * M

    # dataframe to save the results
    results = pd.DataFrame(columns=['dataset','fold','n_ensemble','method_weights','compl_measure','loss_selected','loss_train','loss_test',
                                    'train_acc','test_acc','conf_matr_train', 'conf_matr_test'])

    skf = StratifiedKFold(n_splits=n_cv_splits, random_state=1, shuffle=True)
    fold = 0
    for train_index, test_index in skf.split(X, y):
        fold = fold + 1
        # print(fold)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        loss_train, loss_test, train_acc, test_acc, conf_matr_train, conf_matr_test= gradient_boosting_algorithm(X_train,y_train,X_test,y_test,M,method_weights,
                                                                                                                     loss_f, CM_selected, plot_error)

        fold_v = [fold]*M
        results_dict = {'dataset':dataset_v,'fold':fold_v,'n_ensemble':n_ensemble_v,'method_weights':weights_type,
                        'compl_measure':CM_selected_v,'loss_selected':loss_f_v,
                        'loss_train':loss_train,'loss_test':loss_test,
                        'train_acc':train_acc, 'test_acc':test_acc,
                        'conf_matr_train':conf_matr_train, 'conf_matr_test':conf_matr_test}

        results_aux = pd.DataFrame(results_dict)
        results = pd.concat([results, results_aux])
        results.reset_index(drop=True, inplace=True)

    # Aggregation per fold
    res_agg = aggregation_results_gradientboosting(results)

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

def gradientboosting_all_combinations(path_to_save, dataset, X,y):
    CM_list = ['Hostility', 'kDN', 'DCP', 'TD_U', 'CLD', 'N1', 'N2', 'LSC', 'F1']
    # method_weights_list = ['classic','init_easy','init_hard','init_easy_x2','init_hard_x2',
    #                        'error_w_easy','error_w_hard']
    method_weights_list = ['classic','sample_weight_easy','sample_weight_easy_x2',
                           'sample_weight_hard','sample_weight_hard_x2']
                           #'init_easy','init_hard']
    loss_list = ['log_loss','exponential']
    # method_weights_list = ['classic','error_w_easy','error_w_hard']

    # Para guardar todos los resultados
    results_total = pd.DataFrame(columns=['dataset', 'fold', 'n_ensemble', 'method_weights', 'compl_measure',
                                          'loss_selected', 'loss_train', 'loss_test',
                                         'train_acc', 'test_acc', 'conf_matr_train', 'conf_matr_test'])
    res_agg_total = pd.DataFrame(columns=['dataset', 'n_ensemble', 'method_weights', 'compl_measure',
                                        'loss_selected', 'loss_train_mean', 'loss_test_mean', 'train_acc_mean',
                                        'test_acc_mean', 'loss_train_std', 'loss_test_std', 'train_acc_std',
                                        'test_acc_std', 'conf_matr_train_total', 'conf_matr_test_total'])

    # Algunos parámetros que dejamos fijos
    M = 300 # 10 # 300
    n_cv_splits = 10 # 5 # 10
    plot_error = False


    ## Caso clásico
    method_weights = 'classic'
    CM_selected = 'none'
    for loss_f in loss_list:
        results, res_agg = CV_Gradientboosting(dataset,X,y,M,method_weights,CM_selected, plot_error,n_cv_splits,loss_f)
        results_total = pd.concat([results_total, results])
        res_agg_total = pd.concat([res_agg_total, res_agg])

    # Including complexity
    method_weights_list.remove('classic')
    for method_weights in method_weights_list:
        print(method_weights)
        for CM_selected in CM_list:
            print(CM_selected)
            for loss_f in loss_list:
                results, res_agg = CV_Gradientboosting(dataset,X,y,M,method_weights,CM_selected, plot_error,n_cv_splits,loss_f)

                results_total = pd.concat([results_total, results])
                res_agg_total = pd.concat([res_agg_total, res_agg])



    # To save the results
    os.chdir(path_to_save)
    nombre_csv = 'Results_GB_' + dataset + '.csv'
    nombre_csv_aggr = 'AggregatedResults_GB_' + dataset + '.csv'
    results_total.to_csv(nombre_csv, encoding='utf_8_sig',index=False)
    res_agg_total.to_csv(nombre_csv_aggr, encoding='utf_8_sig', index=False)

    return results_total, res_agg_total




# path_csv = os.chdir(root_path+'/datasets')
path_csv = os.chdir(root_path+'/datasets/nuevos_datos')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv'):
        total_name_list.append(filename)


 #'segment.csv', # da error porque es multiclase, no lo usamos

#total_name_list = ['bands.csv']
# total_name_list = ['teaching_assistant_MH.csv','contraceptive_NL.csv','hill_valley_without_noise_traintest.csv',
#  'glass0.csv','saheart.csv','breast-w.csv','contraceptive_LS.csv', 'yeast1.csv','ilpd.csv',
#     'phoneme.csv','mammographic.csv','contraceptive_NS.csv','bupa.csv','Yeast_CYTvsNUC.csv','ring.csv','titanic.csv',
#  'musk1.csv','spectfheart.csv','arrhythmia_cfs.csv','vertebral_column.csv','profb.csv','sonar.csv',
#  'liver-disorders.csv','steel-plates-fault.csv','credit-g.csv','glass1.csv',
#  'breastcancer.csv', 'diabetes.csv',
#     'diabetic_retinopathy.csv', 'WineQualityRed_5vs6.csv',
#  'teaching_assistant_LM.csv', 'ionosphere.csv', 'bands.csv',
#  'wdbc.csv',
#  'sylvine.csv',
#  'teaching_assistant_LH.csv',
#  'vehicle2.csv',
#  'pima.csv',
#  'spambase.csv',
#  'fri_c0_250_50.csv',
#  'parkinsons.csv',
# 'bodyfat.csv',
#  'banknote_authentication.csv',
#  'chatfield_4.csv'
# ]

# total_name_list = [#'chscase_vine2.csv','chscase_census3.csv','jEdit_4.2_4.3.csv','baskball.csv',
 #'visualizing_ethanol.csv','rabe_97.csv','mbagrade.csv','corral.csv','elusage.csv',
 #'hutsof99_logis.csv',
                   #'magic.csv', # ha petado
 #                   'chscase_census4.csv','triazines.csv',
 # 'chscase_census6.csv','balance-scale.csv','sleuth_ex1714.csv', 'lowbwt.csv',
 # 'sleuth_ex2016.csv','analcatdata_gviolence.csv', 'analcatdata_japansolvent.csv',
 # 'rmftsa_ladata.csv', 'pollution.csv','rmftsa_sleepdata.csv','pm10.csv',
 # 'qualitative-bankruptcy.csv','cmc.csv','jEdit_4.0_4.2.csv','analcatdata_vineyard.csv',
 # 'sleuth_ex1221.csv','lupus.csv','banana.csv','cleve.csv','sleuth_case1201.csv',
 # 'rabe_131.csv','chscase_census2.csv','Australian.csv','disclosure_z.csv','stock.csv',
 # 'zoo.csv','diggle_table_a1.csv','wind_correlations.csv','disclosure_x_bias.csv',
 # 'chscase_census5.csv','rabe_266.csv','sleuth_ex2015.csv','rabe_265.csv',
 # 'disclosure_x_noise.csv','diggle_table_a2.csv','visualizing_environmental.csv',
 # 'vineyard.csv','sleuth_case2002.csv','plasma_retinol.csv','ecoli.csv','chscase_vine1.csv',
 # 'no2.csv','boston.csv','quake.csv','sensory.csv','hutsof99_child_witness.csv',
 # 'visualizing_hamster.csv','pyrim.csv','strikes.csv','witmer_census_1980.csv',
 # 'pwLinear.csv','vinnie.csv','heart-statlog.csv','kc1-binary.csv',
 # 'disclosure_x_tampered.csv',
    #'acute-inflammations.csv', # no ha funcionado
    #'visualizing_galaxy.csv']


path_to_save = root_path + '/Results_GB'
for data_file in total_name_list:
    # os.chdir(root_path + '/datasets')
    os.chdir(root_path + '/datasets/nuevos_datos')
    print(data_file)
    file = data_file
    name_data = data_file[:-4]
    #dataset = name_data
    data = pd.read_csv(file)
    # Get X (features) and y (target)
    X = data.iloc[:, :-1].to_numpy()  # all variables except y
    X = preprocessing.scale(X)
    y = data[['y']].to_numpy().reshape(-1)

    _, _ = gradientboosting_all_combinations(path_to_save, name_data, X, y)


























