### Código para hacer pruebas de complejidad con Gradient Boosting
### Vamos a cambiar los pesos iniciales
### y los pesos del fit

import warnings

from sklearn.neighbors import KNeighborsClassifier

# Suppress only FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

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



from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# 1. Crear datos de clasificación binaria
X, y = make_classification(n_samples=200, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# 2. Crear una "complejidad" falsa: puntaje aleatorio entre 0 y 1
np.random.seed(42)
complexity_score = np.random.rand(len(y_train))  # Simulando medida de complejidad

# 3. Escalarlo entre 0.05 y 0.95 para evitar extremos 0 o 1
# scaled_complexity = 0.05 + 0.9 * complexity_score

# init_logits = np.log((scaled_complexity + 1e-5) / (1 - scaled_complexity + 1e-5))
# init_model = CustomInitModel(initial_predictions=init_logits)
init_model = DecisionTreeClassifier()
init_model.fit(X_train, y_train,sample_weight=complexity_score)

clf = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=1,
    init=init_model
)
clf.fit(X_train, y_train)

# 7. Evaluar
y_pred_proba = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_pred_proba))






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

    elif (method_weights == 'init_easy'):
        # Get complexity measure on train set
        data_train = pd.DataFrame(X_train)
        y_cm = y_train.copy()
        data_train['y'] = y_cm
        data_train.columns = data.columns
        df_measures, _ = all_measures(data_train, False, None, None)
        CM_values = df_measures[CM_selected]
        # Más peso a lo fácil
        ranking_easy = CM_values.rank(method='average', ascending=False)  # more weight to easy
        # Inicializamos con un DT con pesos de la complejidad
        init_model = DecisionTreeClassifier(max_depth=3,random_state=28)
        init_model.fit(X_train, y_train, sample_weight=ranking_easy)

        clf_m = GradientBoostingClassifier(n_estimators=M, learning_rate=0.1, random_state=28,
                                           loss = loss_f,max_depth=1,init=init_model)
        clf_m.fit(X_train, y_train)

    elif (method_weights == 'init_hard'):
        # Get complexity measure on train set
        data_train = pd.DataFrame(X_train)
        y_cm = y_train.copy()
        data_train['y'] = y_cm
        data_train.columns = data.columns
        df_measures, _ = all_measures(data_train, False, None, None)
        CM_values = df_measures[CM_selected]

        ranking_hard = CM_values.rank(method='average', ascending=True)  # more weight to difficult

        init_model = DecisionTreeClassifier(max_depth=3,random_state=28)
        init_model.fit(X_train, y_train, sample_weight=ranking_hard)


        clf_m = GradientBoostingClassifier(n_estimators=M, learning_rate=0.1, random_state=28,
                                           loss = loss_f,max_depth=1,init=init_model)
        clf_m.fit(X_train, y_train)

    elif (method_weights == 'init_knn1'):
        init_model = KNeighborsClassifier(n_neighbors=1)
        init_model.fit(X_train, y_train)

        clf_m = GradientBoostingClassifier(n_estimators=M, learning_rate=0.1, random_state=28,
                                           loss = loss_f,max_depth=1,init=init_model)
        clf_m.fit(X_train, y_train)


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
    method_weights_list = ['classic',#'sample_weight_easy','sample_weight_easy_x2',
                           #'sample_weight_hard','sample_weight_hard_x2']
                           'init_easy','init_hard','init_knn1']
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
    M = 300 # 10 #
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
    nombre_csv = 'Results_GB_init_DT_knn_' + dataset + '.csv'
    nombre_csv_aggr = 'AggregatedResults_GB_init_DT_knn_' + dataset + '.csv'
    results_total.to_csv(nombre_csv, encoding='utf_8_sig',index=False)
    res_agg_total.to_csv(nombre_csv_aggr, encoding='utf_8_sig', index=False)

    return results_total, res_agg_total




path_csv = os.chdir(root_path+'/datasets')
# path_csv = os.chdir(root_path+'/datasets/nuevos_datos')
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

total_name_list = ['bupa.csv','phoneme.csv','pollution.csv','zoo.csv','spambase.csv',
 'sleuth_ex2015.csv','bodyfat.csv','sensory.csv','visualizing_environmental.csv','kc1-binary.csv',
 'steel-plates-fault.csv','glass1.csv','visualizing_ethanol.csv','elusage.csv','rabe_97.csv',
 'teaching_assistant_MH.csv','sleuth_ex1714.csv','chscase_census2.csv','ring.csv','WineQualityRed_5vs6.csv',
 'ecoli.csv','banknote_authentication.csv','contraceptive_NS.csv','hutsof99_logis.csv','diggle_table_a2.csv',
 'chscase_census6.csv','plasma_retinol.csv','vinnie.csv','teaching_assistant_LH.csv','disclosure_x_noise.csv',
 'Yeast_CYTvsNUC.csv', 'wdbc.csv', 'chscase_census5.csv', 'saheart.csv', 'rabe_265.csv',
 'no2.csv', 'vineyard.csv', 'rmftsa_sleepdata.csv', 'balance-scale.csv', 'hill_valley_without_noise_traintest.csv',
 'disclosure_x_tampered.csv', 'titanic.csv', 'diggle_table_a1.csv', 'baskball.csv', 'Australian.csv',
 'cleve.csv', 'spectfheart.csv', 'chscase_vine1.csv', 'banana.csv', 'qualitative-bankruptcy.csv',
 'witmer_census_1980.csv', 'sleuth_ex1221.csv', 'glass0.csv', 'mbagrade.csv', 'quake.csv',
 'sleuth_case1201.csv', 'jEdit_4.0_4.2.csv', 'sylvine.csv', 'breastcancer.csv', 'liver-disorders.csv',
 'rabe_266.csv', 'breast-w.csv', 'diabetic_retinopathy.csv', 'jEdit_4.2_4.3.csv', 'mammographic.csv',
 'contraceptive_NL.csv', 'ilpd.csv', 'analcatdata_vineyard.csv', 'boston.csv', 'cmc.csv', 'ionosphere.csv',
 'pyrim.csv', 'credit-g.csv', 'strikes.csv', 'wind_correlations.csv', 'lowbwt.csv', 'teaching_assistant_LM.csv',
 'profb.csv', 'fri_c0_250_50.csv', 'parkinsons.csv', 'vehicle2.csv', 'vertebral_column.csv',
 'analcatdata_japansolvent.csv','sleuth_ex2016.csv','contraceptive_LS.csv', 'pwLinear.csv', 'yeast1.csv',
 'pima.csv', 'arrhythmia_cfs.csv', 'heart-statlog.csv', 'visualizing_hamster.csv', 'chatfield_4.csv',
 'diabetes.csv', 'hutsof99_child_witness.csv', 'bands.csv', 'rabe_131.csv', 'triazines.csv',
 'rmftsa_ladata.csv', 'chscase_census3.csv', 'chscase_vine2.csv', 'stock.csv', 'corral.csv',
 'disclosure_x_bias.csv', 'visualizing_galaxy.csv', 'analcatdata_gviolence.csv', 'sonar.csv', 'musk1.csv',
 'chscase_census4.csv', 'pm10.csv', 'sleuth_case2002.csv', 'disclosure_z.csv','lupus.csv']


path_to_save = root_path + '/Results_GB'
for data_file in total_name_list:
    os.chdir(root_path + '/datasets')
    # os.chdir(root_path + '/datasets/nuevos_datos')
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
























