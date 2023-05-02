####################### BAGGING CODE WITH COMPLEXITY-DRIVEN SAMPLING ######################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from All_measures import all_measures
import random # for sampling with weights
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


root_path = os.getcwd()

path_csv = os.chdir(root_path+'/datasets')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv'):
        total_name_list.append(filename)


total_name_list = ['Data12.csv']

for data_file in total_name_list:
    os.chdir(root_path + '/datasets')
    print(data_file)
    file = data_file
    name_data = data_file[:-4]
    data = pd.read_csv(file)
    X = data[['x1', 'x2']].to_numpy()
    y = data[['y']].to_numpy()





def get_performance_metrics(y, y_pred):
    acc = accuracy_score(y_pred, y)
    # bacc = balanced_accuracy_score(y, y_pred)
    tn, fn, fp, tp = confusion_matrix(y_pred, y).ravel()
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    # gmean = np.sqrt(recall * specificity)
    # auc = roc_auc_score(y, y_pred)
    # mcc = matthews_corrcoef(y, y_pred)
    if (tp + fp == 0):
        ppv = 0
    else:
        ppv = tp / (tp + fp)
    if (tn + fn == 0):
        npv = 0
    else:
        npv = tn / (tn + fn)
    #
    # gps_num = 4 * ppv * recall * specificity * npv
    # gps_denom = (ppv * recall * npv) + (ppv * recall * specificity) + (npv * specificity * ppv) + (npv * specificity * recall)
    #
    # if (gps_denom == 0):
    #     gps = 0
    # else:
    #     gps = gps_num / gps_denom

    return {
        'acc': acc,
        # 'bacc': bacc,
        'recall': recall,
        'specificity': specificity,
        'ppv':ppv,
        'npv':npv,
        # 'gmean': gmean,
        # 'auc': auc,
        # 'mcc': mcc,
        # 'gps': gps,
        'tn': tn,
        'fn': fn,
        'fp': fp,
        'tp': tp,
    }

# dataframe to save the results
results = pd.DataFrame(columns=['dataset','fold','n_ensemble','weights','confusion_matrix',
                                 'accuracy','info_complexity_dataset','info_complexity_class'])


n_ensembles = 50 # maximum number of ensembles to consider (later we plot and stop when we want)
CM_selected = 'kDN' # selection of the complexity measure to guide the sampling

skf = StratifiedKFold(n_splits=5, random_state=1,shuffle=True)
for train_index, test_index in skf.split(X, y):
    fold = 1
    # print(train_index)
    print(test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # print(X_test)
    # print(y_test)

    # Obtain complexity measures on train set
    data_train = pd.DataFrame(X_train, columns=['x1','x2'])
    data_train['y'] = y_train
    df_measures, _ = all_measures(data_train,False,None, None)

    CM_weights = df_measures[CM_selected] # TRANSFORMAR EN DISTRIBUCION DE PROB
    preds = pd.DataFrame()
    ensemble_preds = pd.DataFrame()
    # i = 0
    for i in range(n_ensembles):

        print(i)
        # Get bootstrap sample following CM_weights
        n_train = len(y_train)
        np.random.seed(0)
        bootstrap_train_sample = random.choices(X_train, weights=CM_weights, k=n_train)
        bootstrap_train_sample = np.array(bootstrap_train_sample) # correct format
        # Save complexity information (class and dataset levels) POR HACER

        # Train DT in bootstrap sample and test y X_test, y_test
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(bootstrap_train_sample, y_train)
        y_pred = clf.predict(X_test)

        if (i<30): # first iterations
            col_name = 'pred_' + str(i)
            preds[col_name] = y_pred  # individual predictions
            # ensemble_preds = pd.DataFrame(y_pred, columns=['pred_0'])
        else:
            col_name = 'pred_'+str(i)
            preds[col_name] = y_pred # individual predictions
            ensemble_preds[col_name] = preds.mode(axis=1)[0] # ensemble prediction with majority voting rule

            y_predicted = ensemble_preds.iloc[:, -1:] # last column
            acc = accuracy_score(y_predicted, y_test)
            conf_matrix = confusion_matrix(y_test, y_predicted)
            results_dict = {'dataset':name_data,'fold':fold, 'n_ensemble':i, 'weights':CM_selected,
                            'confusion_matrix':'conf_matrix', 'accuracy':acc,
                            'info_complexity_dataset':'VALOR',
                            'info_complexity_class':'VALORES'}
            results_aux = pd.DataFrame(results_dict, index=[0])
            results = pd.concat([results,results_aux])

    fold += 1

    # Performance of each ensemble: accuracy and confusion matrix
    # confusion_matrix(y_test, ensemble_preds[col_name]) # PENSAR COMO GUARDAR CONFUSION MATRIX
    # y_predicted = ensemble_preds[col_name]
    # acc = accuracy_score(y_predicted, y_test)
    # get_performance_metrics(y_test, y_predicted)
    # y = y_test


    # Sacaría csv con esta estructura
    # dataset	fold	n_ensemble	weights	confusion_matrix	accuracy	info_complexity_dataset	info_complexity_class












