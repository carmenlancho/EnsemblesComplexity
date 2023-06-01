########################################################################################
########### SCRIPT TO OBTAIN CLASSIFICATION ACCURACY OF THE EASIEST DATASETS ###########
########################################################################################


import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from numpy import *
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.model_selection import train_test_split
import os


root_path = os.getcwd()

def classification_gridsearchCV_easyDatasets(X,y):

    p = len(X[0]) # number of attributes


    # SVM LINEAR
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 1000]}
    svc = SVC(kernel='linear')
    clf = GridSearchCV(SVC(kernel='linear'), param_grid, cv=10, scoring='accuracy', return_train_score=False)
    clf = clf.fit(X, y)
    acc_best_svmlinear = clf.best_score_


    # SVM RBF
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=10, scoring='accuracy', return_train_score=False)
    clf = clf.fit(X, y)
    acc_best_svmrbf = clf.best_score_


    # MLP
    # p = number of attribute
    a = int((p + 2) / 2)
    if (a <= 3):
        param_grid = {'hidden_layer_sizes': [(a,), (a + 1,), (a + 2,), (a + 3,)],
                      'activation': ["logistic", "relu", "tanh"],
                      'learning_rate_init': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
    else:
        param_grid = {'hidden_layer_sizes': [(a - 3,), (a - 2,), (a - 1,), (a,), (a + 1,), (a + 2,), (a + 3,)],
                      'activation': ["logistic", "relu", "tanh"],
                      'learning_rate_init': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}

    clf = GridSearchCV(MLPClassifier(random_state=1, max_iter=800), param_grid, cv=10, scoring='accuracy',
                       return_train_score=False)
    clf = clf.fit(X, y)
    acc_best_mlp = clf.best_score_


    # KNN
    param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17]}
    clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, scoring='accuracy', return_train_score=False)
    clf = clf.fit(X, y)
    acc_best_knn = clf.best_score_


    # DT
    param_grid = {'min_impurity_decrease': [0.1, 0.2, 0.3, 0.4, 0.5],
                  'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    clf = GridSearchCV(DecisionTreeClassifier(random_state=1), param_grid, cv=10, scoring='accuracy',
                       return_train_score=False)
    clf = clf.fit(X, y)
    acc_best_dt = clf.best_score_

    acc_results_dict = {'acc_svmlinear':acc_best_svmlinear,
                   'acc_svmrbf':acc_best_svmrbf,
                   'acc_mlp':acc_best_mlp,
                   'acc_knn':acc_best_knn,
                   'acc_dt':acc_best_dt}
    acc_results = pd.DataFrame([acc_results_dict])

    return acc_results




path_csv = os.chdir(root_path+'/datasets')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv'):
        total_name_list.append(filename)


total_name_list = ['phoneme.csv','spambase.csv', 'ionosphere.csv',
                 'wdbc.csv', 'segment.csv','breast-w.csv',  'banknote_authentication.csv' ]
# total_name_list =  [ 'bupa.csv','hill_valley_without_noise_traintest.csv','contraceptive_NS.csv',
    #              'teaching_assistant_LM.csv','contraceptive_LS.csv','diabetic_retinopathy.csv',
    #              'Yeast_CYTvsNUC.csv','bands.csv','ilpd.csv']
# total_name_list =  [ 'teaching_assistant_LH.csv','teaching_assistant_MH.csv',
    #              'contraceptive_NL.csv','WineQualityRed_5vs6.csv','vertebral_column.csv',
    #              'diabetes.csv','credit-g.csv','arrhythmia_cfs.csv','pima.csv','mammographic.csv',
    #              'titanic.csv','sonar.csv']


path_to_save = root_path+'/Classification_SingleLearner'
total_results = pd.DataFrame()
for data_file in total_name_list:
    os.chdir(root_path + '/datasets')
    print(data_file)
    file = data_file
    name_data = data_file[:-4]
    data = pd.read_csv(file)
    X = data.iloc[:,:-1].to_numpy() # all variables except y
    X = preprocessing.scale(X)
    y = data[['y']].to_numpy().ravel()


    acc_results = classification_gridsearchCV_easyDatasets(X, y)
    acc_results['dataset'] = name_data
    total_results = pd.concat([total_results,acc_results])

# Save csv
nombre_csv = 'ClassificationSingleLearner_EasyDatasets.csv'
# nombre_csv = 'ClassificationSingleLearner_IntermediateDatasets.csv'
# nombre_csv = 'ClassificationSingleLearner_HardDatasets.csv'
total_results.to_csv(nombre_csv, encoding='utf_8_sig', index=True)




