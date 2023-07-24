# -*- coding: utf-8 -*-
"""
Created on June 01 2018

Version 12
Adaboost included

@author: Ahmedul Kabir
"""
import pandas as pd
import numpy as np
import scipy.stats as st
from random import randrange
import time
import random
import itertools
# import winsound # solo funciona en windows
import csv
import os

# import ih_calc   # External file where IH is calculated # así me da error
# from MixedBagging.Code.ih_calc import *  # lo arreglo así

# copio el script entero de ih_calc porque me da error
#####################---------------------------   ih_calc.py   ---------------------------#####################

import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# Create module for concrete IH calculation so that it can be used by other programs
def calculate_concrete_IH(X, y, full, clfList):
    ndata = X.shape[0]
    numClf = len(clfList)  # Num of classifiers
    # knn_clf = KNeighborsClassifier(np.floor(np.sqrt(ndata)/2))  # k = sqrt(n)/2 # esto da error
    n_neigh = np.floor(np.sqrt(ndata) / 2)
    n_neigh_int = n_neigh.astype(np.int32).tolist()
    type(int(n_neigh_int))
    # n_neigh_int = 3
    knn_clf = KNeighborsClassifier(n_neighbors=int(n_neigh_int))  # k = sqrt(n)/2 # le pongo int
    tree_clf = DecisionTreeClassifier(max_depth=5)
    nb_clf = GaussianNB()
    lr_clf = LogisticRegression()
    lda_clf = LinearDiscriminantAnalysis()
    qda_clf = QuadraticDiscriminantAnalysis()

    # Matrix that record misclassification
    misclf_matrix = np.zeros((ndata, numClf))

    # If full = True, perform Leave-one-out cross validation for all classifiers
    if full == True:
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Classifier 0: kNN
            if 0 in clfList:
                knn_clf.fit(X_train, y_train)
                pred_knn = knn_clf.predict(X_test)
                if pred_knn != y_test:
                    misclf_matrix[test_index[0]][0] = 1

            # Classifier 1: Decision Tree
            if 1 in clfList:
                tree_clf.fit(X_train, y_train)
                pred_tree = tree_clf.predict(X_test)
                if pred_tree != y_test:
                    misclf_matrix[test_index[0]][1] = 1

            # Classifier 2: Naive Bayes
            if 2 in clfList:
                nb_clf.fit(X_train, y_train)
                pred_nb = nb_clf.predict(X_test)
                if pred_nb != y_test:
                    misclf_matrix[test_index[0]][2] = 1

            # Classifier 3: Logistic Regression
            if 3 in clfList:
                lr_clf.fit(X_train, y_train)
                pred_lr = lr_clf.predict(X_test)
                if pred_lr != y_test:
                    misclf_matrix[test_index[0]][3] = 1

            # Classifier 4: LDA
            if 4 in clfList:
                lda_clf.fit(X_train, y_train)
                pred_lda = lda_clf.predict(X_test)
                if pred_lda != y_test:
                    misclf_matrix[test_index[0]][4] = 1

            # Classifier 5: QDA
            if 5 in clfList:
                qda_clf.fit(X_train, y_train)
                pred_qda = qda_clf.predict(X_test)
                if pred_qda != y_test:
                    misclf_matrix[test_index[0]][5] = 1

            ih_vector = np.zeros(ndata)
            for i in range(ndata):
                ih_vector[i] = sum(misclf_matrix[i, :]) / numClf

        return ih_vector, misclf_matrix

        # else perform niter by nfolds (default is 5 by 10) fold cross validation
    else:
        niter = 5  # Num of iterations
        nfolds = 10
        misclf = np.zeros((ndata, numClf, niter))  # For each data, misclassif by each classifier on each iteration

        for randseed in range(niter):
            np.random.seed(randseed)
            kf = KFold(n_splits=nfolds, shuffle=True)
            fold = 0
            for tr_idx, test_idx in kf.split(X):

                X_train, X_test = X[tr_idx], X[test_idx]
                y_train, y_test = y[tr_idx], y[test_idx]

                # Classifier 0: kNN
                if 0 in clfList:
                    knn_clf.fit(X_train, y_train)
                    pred_knn = knn_clf.predict(X_test)
                    for i in range(len(test_idx)):
                        if pred_knn[i] != y_test[i]:
                            misclf[test_idx[i]][0][randseed] = 1

                # Classifier 1: Decision Tree
                if 1 in clfList:
                    tree_clf.fit(X_train, y_train)
                    pred_tree = tree_clf.predict(X_test)
                    for i in range(len(test_idx)):
                        if pred_tree[i] != y_test[i]:
                            misclf[test_idx[i]][1][randseed] = 1

                # Classifier 2: Naive Bayes
                if 2 in clfList:
                    nb_clf.fit(X_train, y_train)
                    pred_nb = nb_clf.predict(X_test)
                    for i in range(len(test_idx)):
                        if pred_nb[i] != y_test[i]:
                            misclf[test_idx[i]][2][randseed] = 1

                # Classifier 3: Logistic Regression
                if 3 in clfList:
                    lr_clf.fit(X_train, y_train)
                    pred_lr = lr_clf.predict(X_test)
                    for i in range(len(test_idx)):
                        if pred_lr[i] != y_test[i]:
                            misclf[test_idx[i]][3][randseed] = 1

                # Classifier 4: LDA
                if 4 in clfList:
                    lda_clf.fit(X_train, y_train)
                    pred_lda = lda_clf.predict(X_test)
                    for i in range(len(test_idx)):
                        if pred_lda[i] != y_test[i]:
                            misclf[test_idx[i]][4][randseed] = 1

                # Classifier 5: QDA
                if 5 in clfList:
                    qda_clf.fit(X_train, y_train)
                    pred_qda = qda_clf.predict(X_test)
                    for i in range(len(test_idx)):
                        if pred_qda[i] != y_test[i]:
                            misclf[test_idx[i]][5][randseed] = 1

                fold = fold + 1

        ih_vector = np.zeros(ndata)
        for i in range(ndata):
            ih_vector[i] = sum(sum(misclf[i])) / (
                        numClf * niter)  # Avg of matrix with numClf classifiers and niter iterations

        return ih_vector, misclf

#####################---------------------------   ih_calc.py   ---------------------------#####################

import matplotlib
import matplotlib.pyplot as plt

from sklearn.manifold import MDS
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_hastie_10_2
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_classification
import sklearn.metrics as mt

import matplotlib.pyplot as plt

# Añadido por Carmen
from sklearn.impute import SimpleImputer
import multiprocessing as mp

root_path = os.getcwd()



""" Some global values declared so that they can be easily changed """
# # dataset_list = ['banknote_authentication', 'breast_cancer', 'breast-w', 'cervical_cancer_risk_factors', 'climate_model', 'credit-a', 'heart_disease', 'heart-statlog', 'horse_colic', 'indian_liver_patients', 'ionosphere','pima_indians_diabetes', 'seismic-bumps', 'stroke_normalized', 'voting_records']
# dataset_list = ['banknote_authentication', 'breast_cancer', 'breast-w', 'cervical_cancer_risk_factors', 'climate_model']
# # dataset_list = ['credit-a', 'heart_disease', 'heart-statlog', 'horse_colic', 'indian_liver_patients', 'ionosphere']
# # dataset_list = ['pima_indians_diabetes', 'seismic-bumps', 'stroke_normalized', 'voting_records']
# # dataset_list = ['breast_cancer', 'heart_disease']
# # dataset_list = ['breast-w','heart-statlog']
# dataset_list = ['acute_inflammations', 'acute_nephritis', 'hepatitis', 'kr-vs-kp', 'labor', 'sick']
# dataset_list = ['balance_scale_LR', 'balance_scale_BL', 'balance_scale_BR']
# dataset_list = ['teaching_assistant_LH', 'teaching_assistant_LM', 'teaching_assistant_MH']
# dataset_list = ['titanic']

classifier = 'DT'  # Which classifier to use. DT = Decision Tree, NB = Naive Bayes, KNN = k-nearest neighbors

synthdata = False  # True whe working with synthetic data
vizbags = False  # True when we want to visualize bags
vizdataset = False  # True when we want to visualize dataset with dimensionality reduction
verbose = True  # Whether to print info of each fold

defnumbags = 20  # default number of bags
numruns = 1  # No. of runs (with diff rand seeds) for k-fold evaluation
numfolds = 10  # No. of folds for test set evaluation
numcvruns = 1  # No. of runs for cross validation
numcvfolds = 10  # No. of folds for cross-validation (for model selection)
numsynthruns = 10  # No. of runs for synthetic dataset experiments
clfList = [0, 1, 2, 3]  # Identify which of the classifiers to use for IH calculation. THe numbers associated with each clf can be found in ih_calc

# Mix ratios and hardness quotients (hq) to be tested
mix_ratios = [[0.4, 0.4, 0.2], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1], [0.2, 0.8, 0.0], [0.3, 0.7, 0.0],
              [0.2, 0.4, 0.4], [0.1, 0.6, 0.3], [0.1, 0.7, 0.2], [0.0, 0.8, 0.2], [0.0, 0.7, 0.3],
              [0.2, 0.6, 0.2], [0.1, 0.8, 0.1], [0.0, 1.0, 0.0]]
hqs = [1.0, 2.0, 3.0]

# Hardness intervals for gradually mixed bags to be tested
hardness_intervals = [[-6.0, 6.0], [-5.0, 5.0], [-4.0, 4.0], [-3.0, 3.0],
                      [-4.0, 3.0], [-3.0, 2.0], [-2.0, 1.0],
                      [-3.0, 4.0], [-2.0, 3.0], [-1.0, 2.0],
                      [-4.0, 2.0], [-3.0, 1.0], [-2.0, 0.0],
                      [-2.0, 4.0], [-1.0, 3.0], [0.0, 2.0],
                      [-2.0, 2.0], [-1.0, 1.0], [0.0, 0.0]]

numratios = len(mix_ratios)
numhqs = len(hqs)
""" Parameter values of make_classification defined globally for ease of modification """
nsamples = 500
nfeatures = 10
ninformative = np.round(nfeatures * 0.6)  # 60% of features are informative
nredundant = np.round(nfeatures * 0.1)  # 10% of features are redundant.   Remaining 30% are useless (noise) features
flipy = 0.05
balance = [0.5, 0.5]
# randomstate = 1
classsep = 1.0
nclusperclass = 2

""" Create 'normal' bags: a random subsample from the dataset with replacement """


def subsample(dataset, ratio=1.0):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    # random.seed(10)
    sample_indices = np.random.choice(len(dataset), n_sample)
    for i in range(n_sample):
        index = sample_indices[i]
        sample.append(dataset[index])
    sample_asarray = np.asarray(sample)
    return sample_asarray


""" Create 'hard' bags: a random subsample from the dataset with replacement """
""" ih contains the hardness of each instance in the dataset """


def hard_subsample(dataset, ih, hard_quotient, ratio=1.0):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    # Set weights to be a function of IH by adding a certain percentage of IH (as specified by hard_quotient) to weights
    w = np.ones(len(dataset)) + [hard_quotient * i for i in
                                 ih]  # This way the weights are between 1 and 1.2 if hard_quotient = 0.2 (default)
    weights = w / sum(w)  # Now sum of all weights equal to one

    #  np.random.seed(10)
    sample_indices = np.random.choice(len(dataset), n_sample, p=weights)
    for i in range(n_sample):
        index = sample_indices[i]
        sample.append(dataset[index])
    sample_asarray = np.asarray(sample)
    return sample_asarray


""" Create 'easy' bags: a random subsample from the dataset with replacement """
""" ih contains the hardness of each instance in the dataset. We create 'easiness' as 1 - IH or opposite of hardness """


def easy_subsample(dataset, ih, hard_quotient, ratio=1.0):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    # Set weights to be a function of IH by adding a certain percentage of IH (as specified by hard_quotient) to weights
    easiness = [1 - i for i in ih]
    w = np.ones(len(dataset)) + [hard_quotient * i for i in
                                 easiness]  # This way the weights are between 1 and 1.2 if hard_quotient = 0.2 (default)
    weights = w / sum(w)  # Now sum of all weights equal to one

    #  np.random.seed(10)
    sample_indices = np.random.choice(len(dataset), n_sample, p=weights)
    for i in range(n_sample):
        index = sample_indices[i]
        sample.append(dataset[index])
    sample_asarray = np.asarray(sample)
    return sample_asarray


""" Original Wagging method (Bauer & Kohavi 1999) """


def wagging_weights_gaussian(n):
    #  random.seed(10)  # Arbitrary random seed
    # Set weights to be 1 + random noise with mean at 0 and std. dev. = 2.0
    w = np.ones(n) + np.random.normal(0.0, 2.0, n)
    w[
        w < 0] = 0  # Some weights become less than 0, so they are brought back to just zero and simply disappear (Bauer, Kohavi 1999)
    return w


""" Create weights for wagging """


def create_wagging_weights(num, nbags=defnumbags):
    bag_weights = list()

    for i in range(nbags):
        weights = wagging_weights_gaussian(num)
        bag_weights.append(weights)

    bag_weights_asarray = np.asarray(bag_weights)
    return bag_weights_asarray


""" Make wagging models from weights and test them on the testing set """


def make_wagging_models(bag_weights, clf, X_train, Y_train, X_test, Y_test):
    nbags = bag_weights.shape[0]

    predictions = list()
    predictions_proba = list()

    for i in range(nbags):
        # Fit and predict (hard as well as soft predictions)
        clf.fit(X_train, Y_train, sample_weight=bag_weights[i])
        predicted = clf.predict(X_test)
        # print(predicted.shape)
        predictions.append(predicted)
        pred_proba = clf.predict_proba(X_test)
        predictions_proba.append(pred_proba)

        predictions_asarray = np.asarray(predictions)
        predictions_proba_asarray = np.asarray(predictions_proba)
    return predictions_asarray, predictions_proba_asarray


""" Create regular bootstraps """


def create_regular_bags(trainset, nbags=defnumbags):
    bags = list()

    for i in range(nbags):
        bags.append(subsample(trainset))

    bags_asarray = np.asarray(bags)
    return bags_asarray


""" Create bootstraps (bags) from a given dataset (training set) """
""" Mix ratio: ratio of easy, normal and hard bags (default 0.33, 0.33, 0.33) """
""" hard_quotient: How much oversampling or undersampling to be done """
""" Regular (conventional) bagging is a special case where mix ratio is 0:1:0 """


def create_mixed_bags(trainset, ih, nbags=defnumbags, mix_ratio=[0.33, 0.33, 0.33], hard_quotient=0.5):
    neasy = round(nbags * mix_ratio[0])
    nnormal = round(nbags * mix_ratio[1])
    nhard = round(nbags * mix_ratio[2])

    bags = list()
    # Make easy bags
    for i in range(neasy):  # First nneasy easy bags
        bags.append(easy_subsample(trainset, ih, hard_quotient))
        # Make normal bags
    for i in range(neasy, neasy + nnormal):  # The next nnormal normal bags
        bags.append(subsample(trainset))
    # Make hard bags
    for i in range(neasy + nnormal, neasy + nnormal + nhard):  # The next nhard hard bags
        bags.append(hard_subsample(trainset, ih, hard_quotient))

    bags_asarray = np.asarray(bags)
    return bags_asarray


""" Create bootstraps for gradually-changing mixed bagging """


def create_gradually_mixed_bags(trainset, ih, nbags=defnumbags, low_bag_hardness=0.5, high_bag_hardness=1.5):
    bag_hardness_values = np.linspace(low_bag_hardness, high_bag_hardness,
                                      nbags)  # Divide up the range to find hardness value for each bag

    bags = list()

    for i in range(nbags):  # nbags = len(hardness_values)
        if bag_hardness_values[i] < 0:
            bags.append(easy_subsample(trainset, ih, 0 - bag_hardness_values[
                i]))  # say, bag hardness is -0.3 Thats the same as creating an easy bag with HC =  0.3
        elif bag_hardness_values[i] == 0:
            bags.append(subsample(trainset))
        else:  # if > 1
            bags.append(hard_subsample(trainset, ih, bag_hardness_values[
                i]))  # say, bag_hardness is 0.4. Thats the same as creating a hard bag with HC = 0.4

    bags_asarray = np.asarray(bags)
    return bags_asarray


""" Make scatterplots of each bag """


def visualize_bags(bags):
    nbags = bags.shape[0]
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

    #  mycolor = bags[i][:,2]*(ih + 0.2)
    for i in range(nbags):
        plt.subplot(3, 3, i + 1)  # subplots are numbered 1->n not 0->n

        # We are usually plotting the default 3:3:3 case
        if i < 3:
            bagtype = 'easy'
        elif i < 6:
            bagtype = 'regular'
        else:
            bagtype = 'hard'

        plt.title("Bag %d (%s)" % (i + 1, bagtype), fontsize=10)
        # color (blue/red) = sign: will reflect class, prominence = magnitude: will reflect IH. Added 0.2 so that y*IH
        # doesn't become zero (thus making colors/classes indistinguishable) effectively shifting the IH range from (0,1) to (0.2, 1.2)
        y = bags[i][:, 2]
        y[y == 0] = -1
        ih = bags[i][:, 3]
        plt.scatter(bags[i][:, 0], bags[i][:, 1], c=y * (ih + 0.2), cmap=plt.cm.coolwarm, s=10)
    plt.show()


""" MDS visualization of dataset """


def visualize_dataset(X, y, ih):
    transformer = MDS(n_components=2, max_iter=100, n_init=1)
    fig, plot = plt.subplots()
    fig.set_size_inches(8, 8)
    plt.prism()

    # Blue for low (good) mRS-90, Red for High
    colors = ['blue', 'red']

    X_transformed = transformer.fit_transform(X)
    y[y == 0] = -1
    # color (blue/red) = sign: will reflect class, prominence = magnitude: will reflect IH. Added 0.2 so that y*IH
    # doesn't become zero (thus making colors/classes indistinguishable) effectively shifting the IH range from (0,1) to (0.2, 1.2)
    plot.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y * (ih + 0.2), cmap=plt.cm.coolwarm)
    plot.set_xticks(())
    plot.set_yticks(())

    count = 0;
    plt.tight_layout()
    plt.suptitle("MDS of dataset", fontsize=20)
    # for label , x, y in zip(y_train, X_transformed[:, 0], X_transformed[:, 1]):
    # Lets annotate every 1 out of 20 samples, otherwise graph will be cluttered with anotations
    #  if count % 20 == 0:
    #    plt.annotate(str(int(label)),xy=(x,y), color='black', weight='normal',size=10,bbox=dict(boxstyle="round4,pad=.5", fc="0.8"))
    #  count = count + 1
    # plt.savefig("mnist_pca.png")
    plt.show()


""" Make models from each bag and test them on the testing set """


def make_models(bags, clf, X_test, Y_test):
    nbags = bags.shape[0]
    nfeatures = bags.shape[2] - 2  # because the last two columns are y and ih

    predictions = list()
    predictions_proba = list()

    for i in range(nbags):
        # Fit and predict (hard as well as soft predictions)
        clf.fit(bags[i][:, 0:nfeatures], bags[i][:, nfeatures])
        predicted = clf.predict(X_test)
        # print(predicted.shape)
        predictions.append(predicted)
        pred_proba = clf.predict_proba(X_test)
        predictions_proba.append(pred_proba)

        # Export tree as DOT file for visualization
        #    outfilename = "tree_bag"+ str(i) + ".dot"
        #    export_graphviz(clf, out_file=outfilename)   # tree_bag1.dot, tree_bag2.dot etc.

        predictions_asarray = np.asarray(predictions)
        predictions_proba_asarray = np.asarray(predictions_proba)
    return predictions_asarray, predictions_proba_asarray


""" Calculate final prediction from the predictions of several bag """


def calculate_pred(predictions, predictions_proba):
    ninst = predictions.shape[1]  # No. of instances
    nbags = predictions.shape[0]  # No. of bags

    # hard (binary) predictions
    final_pred = np.zeros((ninst, 1))
    for j in range(ninst):  # for each instance
        count = 0
        for i in range(nbags):  # find prediction in each bag and do majority vote
            if predictions[i, j] == 1:
                count = count + 1
        final_pred[j] = 1 if count > nbags / 2 else 0  # Majority vote

    # soft (with probabilities) predictions
    final_pred_proba = np.zeros((ninst, 1))
    for j in range(ninst):  # for each instance
        s = 0  # sum
        for i in range(nbags):  # find prediction in each bag and take average
            s = s + predictions_proba[i, j, 1]  # Probabilities of class 1
        avg = s / nbags
        final_pred_proba[j] = 1 if avg > 0.5 else 0  # Binarize the predictions

    return final_pred, final_pred_proba


""" MAIN SCRIPT ============================================================="""



def AdaptedMixedBagging(dataset_name):
    classifier = 'DT'  # Which classifier to use. DT = Decision Tree, NB = Naive Bayes, KNN = k-nearest neighbors

    synthdata = False  # True whe working with synthetic data
    vizbags = False  # True when we want to visualize bags
    vizdataset = False  # True when we want to visualize dataset with dimensionality reduction
    verbose = True  # Whether to print info of each fold

    defnumbags = 20  # default number of bags
    numruns = 1  # No. of runs (with diff rand seeds) for k-fold evaluation
    numfolds = 10  # No. of folds for test set evaluation
    numcvruns = 1  # No. of runs for cross validation
    numcvfolds = 10  # No. of folds for cross-validation (for model selection)
    numsynthruns = 10  # No. of runs for synthetic dataset experiments
    clfList = [0, 1, 2,
               3]  # Identify which of the classifiers to use for IH calculation. THe numbers associated with each clf can be found in ih_calc

    # Mix ratios and hardness quotients (hq) to be tested
    mix_ratios = [[0.4, 0.4, 0.2], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1], [0.2, 0.8, 0.0], [0.3, 0.7, 0.0],
                  [0.2, 0.4, 0.4], [0.1, 0.6, 0.3], [0.1, 0.7, 0.2], [0.0, 0.8, 0.2], [0.0, 0.7, 0.3],
                  [0.2, 0.6, 0.2], [0.1, 0.8, 0.1], [0.0, 1.0, 0.0]]
    hqs = [1.0, 2.0, 3.0]

    # Hardness intervals for gradually mixed bags to be tested
    hardness_intervals = [[-6.0, 6.0], [-5.0, 5.0], [-4.0, 4.0], [-3.0, 3.0],
                          [-4.0, 3.0], [-3.0, 2.0], [-2.0, 1.0],
                          [-3.0, 4.0], [-2.0, 3.0], [-1.0, 2.0],
                          [-4.0, 2.0], [-3.0, 1.0], [-2.0, 0.0],
                          [-2.0, 4.0], [-1.0, 3.0], [0.0, 2.0],
                          [-2.0, 2.0], [-1.0, 1.0], [0.0, 0.0]]

    numratios = len(mix_ratios)
    numhqs = len(hqs)
    """ Parameter values of make_classification defined globally for ease of modification """
    nsamples = 500
    nfeatures = 10
    ninformative = np.round(nfeatures * 0.6)  # 60% of features are informative
    nredundant = np.round(
        nfeatures * 0.1)  # 10% of features are redundant.   Remaining 30% are useless (noise) features
    flipy = 0.05
    balance = [0.5, 0.5]
    # randomstate = 1
    classsep = 1.0
    nclusperclass = 2


    # path_to_save = root_path + '/MixedBagging/Adapted_results' # ordenador
    path_to_save = root_path + '/Adapted_results' # server

    verbose = True

    # defnumbags = 20  # default number of bags
    # defnumbags_v = [8, 10]
    defnumbags_v = [1,10,20,30,40,50,60,70,80,90,
                     100,110,120,130,140,150,160,170,180,190,200]

    np.set_printoptions(threshold=np.inf)  # To print the whole array instead of "..." in the middle parts

    datasetcount = 0

    """ Begin loop for each dataset """
    # for dataset_name in os.listdir('MixedBagging/Datasets/'):    # dataset_name now contains the trailing '.csv'
    if (dataset_name.endswith('csv')):

        results_df = pd.DataFrame()  # results for each dataset

        for defnumbags in defnumbags_v:

            # # Create files to write out the prints
            # outfilename = "MixedBagging/outputs/DT_d1/" + dataset_name[:-4] + "_n_trees_" +str(defnumbags) + "_summary.txt"
            # outfile = open(outfilename, "w")
            #
            # csvfilename = "MixedBagging/outputs/DT_d1/" + dataset_name[:-4] + "_n_trees_" +str(defnumbags) + "_details.csv"
            # csvfile = open(csvfilename, 'w')
            # csvwriter = csv.writer(csvfile, delimiter=",", lineterminator='\n')

            # data = pd.read_csv(root_path + '/MixedBagging/Datasets/' + dataset_name)  # ordenador
            data = pd.read_csv(root_path + '/Datasets/' + dataset_name)  # server
            X_mis = data.iloc[:, :-1]  # Columns 0 to end - 1 # cambio ix por iloc
            y = data.iloc[:, -1]  # Last column # cambio ix por iloc

            imp = SimpleImputer(missing_values=999,
                                strategy='most_frequent')  # Impute missing values which are coded as'999' in all datasets
            # imp = preprocessing.Imputer(missing_values=999, strategy='most_frequent', axis=0)  # Impute missing values which are coded as'999' in all datasets
            # preprocessing.Imputer está deprecated, lo cambio por SimpleImputer
            X_unscaled = imp.fit_transform(X_mis)
            X = preprocessing.scale(X_unscaled)  # Scale/normalize the data # NO COMMENT
            # X = X_unscaled
            # '''
            #  X = X.as_matrix()

            #   y = y.as_matrix() # deprecated, lo cambio por to_numpy
            y = y.to_numpy()

            # Initialize arrays to keep results of metrics
            accuracy_base = [0 for i in
                             range(numruns * numfolds)]  # Accuracy of base learner for each fold in each run
            accuracy_reg = [0 for i in
                            range(numruns * numfolds)]  # Accuracy of regular bagging for each fold in each run
            accuracy_wag = [0 for i in range(numruns * numfolds)]  # Accuracy of wagging for each fold in each run
            accuracy_boost = [0 for i in range(numruns * numfolds)]  # Accuracy of wagging for each fold in each run
            accuracy_mix = [0 for i in
                            range(numruns * numfolds)]  # Accuracy of mixed bagging for each fold in each run
            accuracy_grad = [0 for i in range(
                numruns * numfolds)]  # Accuracy of gradually mixed bagging for each fold in each run

            f1_base = [0 for i in range(numruns * numfolds)]
            f1_reg = [0 for i in range(numruns * numfolds)]
            f1_wag = [0 for i in range(numruns * numfolds)]
            f1_boost = [0 for i in range(numruns * numfolds)]
            f1_mix = [0 for i in range(numruns * numfolds)]
            f1_grad = [0 for i in range(numruns * numfolds)]

            auc_base = [0 for i in range(numruns * numfolds)]
            auc_reg = [0 for i in range(numruns * numfolds)]
            auc_wag = [0 for i in range(numruns * numfolds)]
            auc_boost = [0 for i in range(numruns * numfolds)]
            auc_mix = [0 for i in range(numruns * numfolds)]
            auc_grad = [0 for i in range(numruns * numfolds)]

            list_cf_mix_acc = [None] * numfolds
            list_cf_mix_auc = [None] * numfolds
            list_cf_mix_f1 = [None] * numfolds

            list_cf_gr_acc = [None] * numfolds
            list_cf_gr_auc = [None] * numfolds
            list_cf_gr_f1 = [None] * numfolds

            list_cf_base = [None] * numfolds
            list_cf_wagging = [None] * numfolds
            list_cf_boosting = [None] * numfolds

            """ Loop for one run of k-fold  evaluation """
            for run in range(numruns):

                print("Run %d of %d" % (run + 1, numruns))
                fold = 0

                # Split into kfold folds for evaluation
                skf = StratifiedKFold(n_splits=numfolds, shuffle=True,
                                      random_state=run)  # A different random state for each run

                ''' Loop for evaluation of test set '''
                for tr_val_idx, test_idx in skf.split(X, y):
                    X_tr_val, X_test = X[tr_val_idx], X[test_idx]
                    Y_tr_val, Y_test = y[tr_val_idx], y[test_idx]

                    print("\n Still running dataset %s ........" % dataset_name)
                    print("\nFold %d of %d" % (fold + 1, numfolds))

                    # Initialize some arrays to store cv results for each combo/interval in each cv fold
                    # Basically its a (numcvruns*numcvfold) * numCombo array (numCombo = numhqs*numratios or numCombo = len(hardness_intervals))
                    cv_acc_mx = np.zeros(
                        (numcvruns * numcvfolds, numhqs * numratios))  # one for each combo within this cv fold
                    cv_f1_mx = np.zeros((numcvruns * numcvfolds, numhqs * numratios))
                    cv_auc_mx = np.zeros((numcvruns * numcvfolds, numhqs * numratios))
                    cv_confmatrix_mx = np.zeros((numcvruns * numcvfolds, numhqs * numratios))

                    cv_acc_gr = np.zeros(
                        (numcvruns * numcvfolds,
                         len(hardness_intervals)))  # one for each interval within this cv fold
                    cv_f1_gr = np.zeros((numcvruns * numcvfolds, len(hardness_intervals)))
                    cv_auc_gr = np.zeros((numcvruns * numcvfolds, len(hardness_intervals)))
                    cv_confmatrix_gr = np.zeros((numcvruns * numcvfolds, numhqs * numratios))

                    """ Loop for one run of cross validation """
                    for cvrun in range(numcvruns):

                        print("\nCV run %d of %d" % (cvrun + 1, numcvruns))
                        cvfold = 0

                        skf2 = StratifiedKFold(n_splits=numcvfolds, shuffle=True, random_state=cvrun)

                        # Loop for cross validation for model selection
                        for tr_idx, val_idx in skf2.split(X_tr_val, Y_tr_val):
                            X_train, X_validate = X_tr_val[tr_idx], X_tr_val[val_idx]
                            Y_train, Y_validate = Y_tr_val[tr_idx], Y_tr_val[val_idx]

                            #             print("CV Fold %d of %d" % (cvfold+1, numcvfolds))

                            # Calculate IH of training set from external file
                            # ih_tr, misclf = ih_calc.calculate_concrete_IH(X_train, Y_train, full=False, clfList=clfList)
                            ih_tr, misclf = calculate_concrete_IH(X_train, Y_train, full=False,
                                                                  clfList=clfList)  # le quito ih_calc, la carga es diferente

                            # Join the X, Y and IH of training set to make one full dataset for easier manipulation
                            Y_train_reshaped = Y_train.reshape((len(Y_train), 1))
                            ih_tr_reshaped = ih_tr.reshape((len(ih_tr), 1))
                            trainset = np.concatenate((X_train, Y_train_reshaped, ih_tr_reshaped),
                                                      axis=1)  # trainset includes both X and Y

                            mixed_bags = list()
                            for hq, mx in itertools.product(hqs, mix_ratios):
                                mixed_bags.append(create_mixed_bags(trainset, ih_tr, mix_ratio=mx,
                                                                    hard_quotient=hq))  # Mixed bags with different mixed ratios

                            grad_bags = list()
                            for h_i in hardness_intervals:
                                grad_bags.append(
                                    create_gradually_mixed_bags(trainset, ih_tr, low_bag_hardness=h_i[0],
                                                                high_bag_hardness=h_i[
                                                                    1]))  # GRadually mixed bags with different intervals

                            # Visualize bags when necessary. Code just has mixed bags, but can try others
                            if vizbags == True:
                                visualize_bags(mixed_bags[2 * numratios])  # MR = 3:3:3, HQ = 0.75

                            #               print("Building models ...")

                            # Define classifier models
                            if classifier == 'DT':
                                clf = DecisionTreeClassifier(
                                    random_state=cvrun * numcvfolds + cvfold)  # Best depth known for that dataset. Random state uniquely identifying this cvfold of this cvrun
                            elif classifier == 'KNN':
                                clf = KNeighborsClassifier(5)  # k = sqrt(n)/2
                            else:  # NB
                                # clf = GaussianNB()
                                clf = BernoulliNB()

                            mix_predictions = list()
                            mix_predictions_proba = list()
                            mix_final_pred = list()
                            mix_final_pred_proba = list()

                            grad_predictions = list()
                            grad_predictions_proba = list()
                            grad_final_pred = list()
                            grad_final_pred_proba = list()

                            # For mixed bagging, evaluate the validation set
                            for i in range(numhqs * numratios):  # For each mix_ratio for each hq
                                pred, pred_proba = make_models(mixed_bags[i], clf, X_validate, Y_validate)
                                mix_predictions.append(pred)
                                mix_predictions_proba.append(pred_proba)
                                fin_pred, fin_pred_proba = calculate_pred(pred, pred_proba)
                                mix_final_pred.append(fin_pred)
                                mix_final_pred_proba.append(fin_pred_proba)

                                # For gradually mixed bagging, evaluate the validation set
                            for i in range(len(hardness_intervals)):
                                gpred, gpred_proba = make_models(grad_bags[i], clf, X_validate,
                                                                 Y_validate)  # gpred = pred for gradually mixed bags
                                grad_predictions.append(gpred)
                                grad_predictions_proba.append(gpred_proba)
                                gfin_pred, gfin_pred_proba = calculate_pred(gpred, gpred_proba)
                                grad_final_pred.append(
                                    gfin_pred)  # gfin_pred = final prediction for gradually mixed bags
                                grad_final_pred_proba.append(gfin_pred_proba)

                            #            print("Evaluating mixed bagging models ...")

                            # Check all MR-HQ combos for mixed bags
                            for i in range(numhqs * numratios):
                                #                            cv_acc_mx[cvrun*numcvfolds + cvfold][i] = mt.accuracy_score(Y_validate, mix_final_pred_proba[i])
                                #                            cv_f1_mx[cvrun*numcvfolds + cvfold][i] = mt.f1_score(Y_validate, mix_final_pred_proba[i], average='weighted')
                                #                            cv_auc_mx[cvrun*numcvfolds + cvfold][i] = mt.roc_auc_score(Y_validate, mix_final_pred_proba[i])
                                cv_acc_mx[cvrun * numcvfolds + cvfold][i] = mt.accuracy_score(Y_validate,
                                                                                              mix_final_pred[i])
                                cv_f1_mx[cvrun * numcvfolds + cvfold][i] = mt.f1_score(Y_validate,
                                                                                       mix_final_pred[i],
                                                                                       average='weighted')
                                cv_auc_mx[cvrun * numcvfolds + cvfold][i] = mt.roc_auc_score(Y_validate,
                                                                                             mix_final_pred[i])
                                # cv_confmatrix_mx[cvrun * numcvfolds + cvfold][i] = np.array(mt.confusion_matrix(Y_validate,mix_final_pred[i]).tolist())
                            # Check all hardness intervals for gradually mixed bags
                            for i in range(len(hardness_intervals)):
                                #                            cv_acc_gr[cvrun*numcvfolds + cvfold][i] = mt.accuracy_score(Y_validate, grad_final_pred_proba[i])
                                #                            cv_f1_gr[cvrun*numcvfolds + cvfold][i] = mt.f1_score(Y_validate, grad_final_pred_proba[i], average='weighted')
                                #                            cv_auc_gr[cvrun*numcvfolds + cvfold][i] = mt.roc_auc_score(Y_validate, grad_final_pred_proba[i])
                                cv_acc_gr[cvrun * numcvfolds + cvfold][i] = mt.accuracy_score(Y_validate,
                                                                                              grad_final_pred[i])
                                cv_f1_gr[cvrun * numcvfolds + cvfold][i] = mt.f1_score(Y_validate,
                                                                                       grad_final_pred[i],
                                                                                       average='weighted')
                                cv_auc_gr[cvrun * numcvfolds + cvfold][i] = mt.roc_auc_score(Y_validate,
                                                                                             grad_final_pred[i])
                                # cv_confmatrix_gr[cvrun * numcvfolds + cvfold][i] = mt.confusion_matrix(Y_validate,
                                #                                                             grad_final_pred[i])

                            cvfold = cvfold + 1
                        """ End of one run of cv """
                    """ End of fold for cross validation for model selection """

                    """ We now make models for each method and evaluate on test set """

                    # Use the evaluation matrices to find best parameters for mixed bags
                    val_acc_mx = np.mean(cv_acc_mx,
                                         axis=0)  # Take mean validation accuracy of each combo by taking mean of each column
                    val_f1_mx = np.mean(cv_f1_mx, axis=0)  # val = validation
                    val_auc_mx = np.mean(cv_auc_mx, axis=0)
                    # val_confmatrix_mx = np.mean(cv_confmatrix_mx, axis=0)

                    val_acc_gr = np.mean(cv_acc_gr, axis=0)
                    val_f1_gr = np.mean(cv_f1_gr, axis=0)
                    val_auc_gr = np.mean(cv_auc_gr, axis=0)
                    # val_confmatrix_gr = np.mean(cv_confmatrix_gr, axis=0)

                    # Write these values to fie for later inspection

                    # csvwriter.writerows([np.round(val_acc_mx, 3)])
                    # csvwriter.writerows([np.round(val_f1_mx, 3)])
                    # csvwriter.writerows([np.round(val_auc_mx, 3)])
                    # csvwriter.writerow([])
                    # csvwriter.writerows([np.round(val_acc_gr, 3)])
                    # csvwriter.writerows([np.round(val_f1_gr, 3)])
                    # csvwriter.writerows([np.round(val_auc_gr, 3)])
                    # csvwriter.writerow([])

                    # Find indices of best combo/interval
                    idx_max_acc_mx = np.argmax(val_acc_mx)  # Index of max accuracy in mixed bags
                    idx_max_f1_mx = np.argmax(val_f1_mx)
                    idx_max_auc_mx = np.argmax(val_auc_mx)

                    idx_max_acc_gr = np.argmax(val_acc_gr)  # Index of max accuracy in grad bags
                    idx_max_f1_gr = np.argmax(val_f1_gr)
                    idx_max_auc_gr = np.argmax(val_auc_gr)

                    # Calculate IH of whole data (training + vaidation) from external file
                    # ih, misclf = ih_calc.calculate_concrete_IH(X_tr_val, Y_tr_val, full=False, clfList=clfList)
                    ih, misclf = calculate_concrete_IH(X_tr_val, Y_tr_val, full=False,
                                                       clfList=clfList)  # le quito ih_calc

                    # Visualize when needed
                    if vizdataset == True:
                        visualize_dataset(X_tr_val, Y_tr_val, ih)

                    # Join the X, Y and IH of training+validation set to make one full dataset for easier manipulation
                    Y_tr_val_reshaped = Y_tr_val.reshape((len(Y_tr_val), 1))
                    ih_reshaped = ih.reshape((len(ih), 1))
                    trainvalset = np.concatenate((X_tr_val, Y_tr_val_reshaped, ih_reshaped),
                                                 axis=1)  # trainset includes both X and Y

                    # Create bags for everyone. For mixed bags, use the combo/interval we found to be best according to validation set
                    # Regular bagging
                    reg_bags = create_regular_bags(trainvalset)

                    # Wagging. Its slightly different because we get a bag of weights rather than instances
                    wag_weights_bags = create_wagging_weights(len(X_tr_val))

                    # Mixed bags
                    # idx_max_xxx_mx now has index of best mixture in terms of acc, f1, auc. But idx contains info of both HQ and mix_ratio.
                    # Example: numratios = 10, numhqs = 3, so idx= 24 actually means hq = 24/10 = 2 and mix_ratio = 24 % 10 = 4
                    mixed_bags_acc = create_mixed_bags(trainvalset, ih,
                                                       mix_ratio=mix_ratios[idx_max_acc_mx % numratios],
                                                       hard_quotient=hqs[
                                                           idx_max_acc_mx // numratios])  # Mixed bags with different mixed ratios
                    mixed_bags_f1 = create_mixed_bags(trainvalset, ih,
                                                      mix_ratio=mix_ratios[idx_max_f1_mx % numratios],
                                                      hard_quotient=hqs[
                                                          idx_max_f1_mx // numratios])  # Mixed bags with different mixed ratios
                    mixed_bags_auc = create_mixed_bags(trainvalset, ih,
                                                       mix_ratio=mix_ratios[idx_max_auc_mx % numratios],
                                                       hard_quotient=hqs[
                                                           idx_max_auc_mx // numratios])  # Mixed bags with different mixed ratios

                    # if verbose == True:  # Print when required
                    #     outfile.write("\nFor run %d fold %d:" % (run + 1, fold + 1))
                    #     outfile.write("\t%s  %s" % (
                    #     mix_ratios[idx_max_acc_mx % numratios], hqs[idx_max_acc_mx // numratios]))  # integer division
                    #     outfile.write(
                    #         "\t%s  %s" % (mix_ratios[idx_max_f1_mx % numratios], hqs[idx_max_f1_mx // numratios]))
                    #     outfile.write(
                    #         "\t%s  %s" % (mix_ratios[idx_max_auc_mx % numratios], hqs[idx_max_auc_mx // numratios]))

                    # Grad bags
                    grad_bags_acc = create_gradually_mixed_bags(trainvalset, ih,
                                                                low_bag_hardness=hardness_intervals[idx_max_acc_gr][
                                                                    0],
                                                                high_bag_hardness=
                                                                hardness_intervals[idx_max_acc_gr][1])
                    grad_bags_f1 = create_gradually_mixed_bags(trainvalset, ih,
                                                               low_bag_hardness=hardness_intervals[idx_max_f1_gr][
                                                                   0],
                                                               high_bag_hardness=hardness_intervals[idx_max_f1_gr][
                                                                   1])
                    grad_bags_auc = create_gradually_mixed_bags(trainvalset, ih,
                                                                low_bag_hardness=hardness_intervals[idx_max_auc_gr][
                                                                    0],
                                                                high_bag_hardness=
                                                                hardness_intervals[idx_max_auc_gr][1])

                    # if verbose == True:
                    #     #   outfile.write("\nFor run %d fold %d:" % (run + 1, fold + 1))
                    #     outfile.write("\t\t%s" % hardness_intervals[idx_max_acc_gr])
                    #     outfile.write("\t%s" % hardness_intervals[idx_max_f1_gr])
                    #     outfile.write("\t%s" % hardness_intervals[idx_max_auc_gr])

                    # Visualize bags when necessary. Code just has mixed bags, but can try others
                    if vizbags == True:
                        visualize_bags(mixed_bags[2 * numratios])  # MR = 3:3:3, HQ = 0.75

                    #            print("Building models ...")

                    # Build classifier models for each bag
                    if classifier == 'DT':
                        # clf2 = DecisionTreeClassifier(max_depth=1, random_state=run * numfolds + fold) # original
                        clf2 = DecisionTreeClassifier(random_state=run * numfolds + fold)  # unpruned DT

                    elif classifier == 'KNN':
                        clf2 = KNeighborsClassifier(5)  # k = sqrt(n)/2
                    else:  # NB
                        # clf2 = GaussianNB()
                        clf2 = BernoulliNB()
                    # clf = KNeighborsClassifier(np.floor(np.sqrt(nsamples)/2))  # k = sqrt(n)/2
                    # clf = GaussianNB()

                    # Create and evaluate a base classifier (NO bagging, just the base learner)
                    # clf_base = DecisionTreeClassifier(max_depth=1,
                    #                                   random_state=run * numfolds + fold)  # random state uniquely identifying this fold of this run ORIGINAL
                    clf_base = DecisionTreeClassifier(
                        random_state=run * numfolds + fold)  # random state uniquely identifying this fold of this run # unpruned DT
                    clf_base.fit(X_tr_val, Y_tr_val)
                    base_predictions = clf_base.predict(X_test)
                    #     csvwriter.writerows([base_predictions])

                    accuracy_base[run * numfolds + fold] = mt.accuracy_score(Y_test,
                                                                             base_predictions)  # all folds in all runs are being kept in aa 1-D array. So (run*numfolds + fold) gives index of thye fold within the array
                    f1_base[run * numfolds + fold] = mt.f1_score(Y_test, base_predictions, average='weighted')
                    auc_base[run * numfolds + fold] = mt.roc_auc_score(Y_test, base_predictions)
                    list_cf_base[fold] = mt.confusion_matrix(Y_test, base_predictions)

                    res_dict_base_acc = {'dataset': dataset_name, 'CV_fold': fold, 'n_trees': defnumbags,
                                         'model': 'base', 'parameters': 'None',
                                         'perf_measure': 'acc', 'perf_value': accuracy_base[run * numfolds + fold],
                                         'Conf_Matrix': [list_cf_base[fold]]}
                    res_dict_base_f1 = {'dataset': dataset_name, 'CV_fold': fold, 'n_trees': defnumbags,
                                        'model': 'base', 'parameters': 'None',
                                        'perf_measure': 'f1', 'perf_value': f1_base[run * numfolds + fold],
                                        'Conf_Matrix': [list_cf_base[fold]]}
                    res_dict_base_auc = {'dataset': dataset_name, 'CV_fold': fold, 'n_trees': defnumbags,
                                         'model': 'base', 'parameters': 'None',
                                         'perf_measure': 'auc', 'perf_value': auc_base[run * numfolds + fold],
                                         'Conf_Matrix': [list_cf_base[fold]]}

                    # Create and evaluate regular bagging
                    reg_predictions, reg_predictions_proba = make_models(reg_bags, clf2, X_test,
                                                                         Y_test)  # Reg = regular, Proba = probabilities
                    reg_final_pred, reg_final_pred_proba = calculate_pred(reg_predictions, reg_predictions_proba)
                    #      csvwriter.writerows([reg_final_pred_proba])

                    accuracy_reg[run * numfolds + fold] = mt.accuracy_score(Y_test, reg_final_pred)
                    f1_reg[run * numfolds + fold] = mt.f1_score(Y_test, reg_final_pred, average='weighted')
                    auc_reg[run * numfolds + fold] = mt.roc_auc_score(Y_test, reg_final_pred)
                    ## este no me hace falta porque ya lo tengo por otro lado

                    # Create and evaluate wagging. Uses a diff function for making models
                    wag_predictions, wag_predictions_proba = make_wagging_models(wag_weights_bags, clf2, X_tr_val,
                                                                                 Y_tr_val, X_test,
                                                                                 Y_test)  # wag = wagging, Proba = probabilities
                    wag_final_pred, wag_final_pred_proba = calculate_pred(wag_predictions, wag_predictions_proba)
                    #      csvwriter.writerows([wag_final_pred_proba])

                    accuracy_wag[run * numfolds + fold] = mt.accuracy_score(Y_test, wag_final_pred)
                    f1_wag[run * numfolds + fold] = mt.f1_score(Y_test, wag_final_pred, average='weighted')
                    auc_wag[run * numfolds + fold] = mt.roc_auc_score(Y_test, wag_final_pred)
                    list_cf_wagging[fold] = mt.confusion_matrix(Y_test, wag_final_pred)

                    res_dict_wagging_acc = {'dataset': dataset_name, 'CV_fold': fold, 'n_trees': defnumbags,
                                            'model': 'wagging', 'parameters': 'None',
                                            'perf_measure': 'acc',
                                            'perf_value': accuracy_wag[run * numfolds + fold],
                                            'Conf_Matrix': [list_cf_wagging[fold]]}
                    res_dict_wagging_f1 = {'dataset': dataset_name, 'CV_fold': fold, 'n_trees': defnumbags,
                                           'model': 'wagging', 'parameters': 'None',
                                           'perf_measure': 'f1', 'perf_value': f1_wag[run * numfolds + fold],
                                           'Conf_Matrix': [list_cf_wagging[fold]]}
                    res_dict_wagging_auc = {'dataset': dataset_name, 'CV_fold': fold, 'n_trees': defnumbags,
                                            'model': 'wagging', 'parameters': 'None',
                                            'perf_measure': 'auc', 'perf_value': auc_wag[run * numfolds + fold],
                                            'Conf_Matrix': [list_cf_wagging[fold]]}

                    # Create and evaluate boosting
                    boost = AdaBoostClassifier(base_estimator=clf2, n_estimators=10,
                                               random_state=run * numfolds + fold)
                    boost.fit(X_tr_val, Y_tr_val)
                    boost_final_pred = boost.predict(X_test)
                    #      csvwriter.writerows([reg_final_pred_proba])

                    accuracy_boost[run * numfolds + fold] = mt.accuracy_score(Y_test, boost_final_pred)
                    f1_boost[run * numfolds + fold] = mt.f1_score(Y_test, boost_final_pred, average='weighted')
                    auc_boost[run * numfolds + fold] = mt.roc_auc_score(Y_test, boost_final_pred)
                    list_cf_boosting[fold] = mt.confusion_matrix(Y_test, boost_final_pred)

                    res_dict_boosting_acc = {'dataset': dataset_name, 'CV_fold': fold, 'n_trees': defnumbags,
                                             'model': 'AdaBoost', 'parameters': 'None',
                                             'perf_measure': 'acc',
                                             'perf_value': accuracy_boost[run * numfolds + fold],
                                             'Conf_Matrix': [list_cf_boosting[fold]]}
                    res_dict_boosting_f1 = {'dataset': dataset_name, 'CV_fold': fold, 'n_trees': defnumbags,
                                            'model': 'AdaBoost', 'parameters': 'None',
                                            'perf_measure': 'f1', 'perf_value': f1_boost[run * numfolds + fold],
                                            'Conf_Matrix': [list_cf_boosting[fold]]}
                    res_dict_boosting_auc = {'dataset': dataset_name, 'CV_fold': fold, 'n_trees': defnumbags,
                                             'model': 'AdaBoost', 'parameters': 'None',
                                             'perf_measure': 'auc', 'perf_value': auc_boost[run * numfolds + fold],
                                             'Conf_Matrix': [list_cf_boosting[fold]]}

                    # Create and evaluate mixed bagging
                    mix_predictions_acc, mix_predictions_proba_acc = make_models(mixed_bags_acc, clf2, X_test,
                                                                                 Y_test)  # Mix = Mixed bags, Proba = probabilities
                    mix_final_pred_acc, mix_final_pred_proba_acc = calculate_pred(mix_predictions_acc,
                                                                                  mix_predictions_proba_acc)
                    #      csvwriter.writerows([mix_final_pred_proba_acc])

                    mix_predictions_f1, mix_predictions_proba_f1 = make_models(mixed_bags_f1, clf2, X_test, Y_test)
                    mix_final_pred_f1, mix_final_pred_proba_f1 = calculate_pred(mix_predictions_f1,
                                                                                mix_predictions_proba_f1)

                    mix_predictions_auc, mix_predictions_proba_auc = make_models(mixed_bags_auc, clf2, X_test,
                                                                                 Y_test)
                    mix_final_pred_auc, mix_final_pred_proba_auc = calculate_pred(mix_predictions_auc,
                                                                                  mix_predictions_proba_auc)

                    accuracy_mix[run * numfolds + fold] = mt.accuracy_score(Y_test, mix_final_pred_acc)
                    f1_mix[run * numfolds + fold] = mt.f1_score(Y_test, mix_final_pred_f1, average='weighted')
                    auc_mix[run * numfolds + fold] = mt.roc_auc_score(Y_test, mix_final_pred_auc)
                    list_cf_mix_auc[fold] = mt.confusion_matrix(Y_test, mix_final_pred_auc)
                    list_cf_mix_acc[fold] = mt.confusion_matrix(Y_test, mix_final_pred_acc)
                    list_cf_mix_f1[fold] = mt.confusion_matrix(Y_test, mix_final_pred_f1)

                    res_dict_mix_acc = {'dataset': dataset_name, 'CV_fold': fold, 'n_trees': defnumbags,
                                        'model': 'Grouped_Mixed_Bagging',
                                        'parameters': [mix_ratios[idx_max_acc_mx % numratios],
                                                       hqs[idx_max_acc_mx // numratios]],
                                        'perf_measure': 'acc', 'perf_value': accuracy_mix[run * numfolds + fold],
                                        'Conf_Matrix': [list_cf_mix_acc[fold]]}
                    res_dict_mix_f1 = {'dataset': dataset_name, 'CV_fold': fold, 'n_trees': defnumbags,
                                       'model': 'Grouped_Mixed_Bagging',
                                       'parameters': [mix_ratios[idx_max_f1_mx % numratios],
                                                      hqs[idx_max_f1_mx // numratios]],
                                       'perf_measure': 'f1', 'perf_value': f1_mix[run * numfolds + fold],
                                       'Conf_Matrix': [list_cf_mix_f1[fold]]}
                    res_dict_mix_auc = {'dataset': dataset_name, 'CV_fold': fold, 'n_trees': defnumbags,
                                        'model': 'Grouped_Mixed_Bagging',
                                        'parameters': [mix_ratios[idx_max_auc_mx % numratios],
                                                       hqs[idx_max_auc_mx // numratios]],
                                        'perf_measure': 'auc', 'perf_value': auc_mix[run * numfolds + fold],
                                        'Conf_Matrix': [list_cf_mix_auc[fold]]}

                    # Create and evaluate gradually mixed bagging
                    grad_predictions_acc, grad_predictions_proba_acc = make_models(grad_bags_acc, clf2, X_test,
                                                                                   Y_test)  # grad = gradually graded bags, Proba = probabilities
                    grad_final_pred_acc, grad_final_pred_proba_acc = calculate_pred(grad_predictions_acc,
                                                                                    grad_predictions_proba_acc)
                    #      csvwriter.writerows([grad_final_pred_proba_acc])

                    grad_predictions_f1, grad_predictions_proba_f1 = make_models(grad_bags_f1, clf2, X_test, Y_test)
                    grad_final_pred_f1, grad_final_pred_proba_f1 = calculate_pred(grad_predictions_f1,
                                                                                  grad_predictions_proba_f1)

                    grad_predictions_auc, grad_predictions_proba_auc = make_models(grad_bags_auc, clf2, X_test,
                                                                                   Y_test)
                    grad_final_pred_auc, grad_final_pred_proba_auc = calculate_pred(grad_predictions_auc,
                                                                                    grad_predictions_proba_auc)

                    accuracy_grad[run * numfolds + fold] = mt.accuracy_score(Y_test, grad_final_pred_acc)
                    f1_grad[run * numfolds + fold] = mt.f1_score(Y_test, grad_final_pred_f1, average='weighted')
                    auc_grad[run * numfolds + fold] = mt.roc_auc_score(Y_test, grad_final_pred_auc)
                    list_cf_gr_auc[fold] = mt.confusion_matrix(Y_test, grad_final_pred_auc)
                    list_cf_gr_acc[fold] = mt.confusion_matrix(Y_test, grad_final_pred_acc)
                    list_cf_gr_f1[fold] = mt.confusion_matrix(Y_test, grad_final_pred_f1)

                    res_dict_gr_acc = {'dataset': dataset_name, 'CV_fold': fold, 'n_trees': defnumbags,
                                       'model': 'Incremental_Mixed_Bagging',
                                       'parameters': [hardness_intervals[idx_max_acc_gr]],
                                       'perf_measure': 'acc', 'perf_value': accuracy_grad[run * numfolds + fold],
                                       'Conf_Matrix': [list_cf_gr_acc[fold]]}
                    res_dict_gr_f1 = {'dataset': dataset_name, 'CV_fold': fold, 'n_trees': defnumbags,
                                      'model': 'Incremental_Mixed_Bagging',
                                      'parameters': [hardness_intervals[idx_max_f1_gr]],
                                      'perf_measure': 'f1', 'perf_value': f1_grad[run * numfolds + fold],
                                      'Conf_Matrix': [list_cf_gr_f1[fold]]}
                    res_dict_gr_auc = {'dataset': dataset_name, 'CV_fold': fold, 'n_trees': defnumbags,
                                       'model': 'Incremental_Mixed_Bagging',
                                       'parameters': [hardness_intervals[idx_max_auc_gr]],
                                       'perf_measure': 'auc', 'perf_value': auc_grad[run * numfolds + fold],
                                       'Conf_Matrix': [list_cf_gr_auc[fold]]}

                    #                if verbose == True:
                    #                    print("Base acc(fold): %s" % accuracy_base[run*numfolds + fold])
                    #                    print("Reg acc(fold): %s" % accuracy_reg[run*numfolds + fold])
                    #                    print("PyBag acc(fold): %s" % accuracy_pybag[run*numfolds + fold])
                    #                    print("Wag acc(fold): %s" % accuracy_wag[run*numfolds + fold])
                    #                    print("Mix acc(fold): %s" % accuracy_mix[run*numfolds + fold])
                    #                    print("Grad acc(fold): %s" % accuracy_grad[run*numfolds + fold])

                    res_dicts = [res_dict_base_acc, res_dict_base_f1, res_dict_base_auc,
                                 res_dict_wagging_acc, res_dict_wagging_f1, res_dict_wagging_auc,
                                 res_dict_boosting_acc, res_dict_boosting_f1, res_dict_boosting_auc,
                                 res_dict_mix_acc, res_dict_mix_f1, res_dict_mix_auc,
                                 res_dict_gr_acc, res_dict_gr_f1, res_dict_gr_auc]

                    res_df = pd.DataFrame.from_dict(res_dicts)
                    results_df = pd.concat([results_df, res_df])
                    # To save the results
                    os.chdir(path_to_save)
                    nombre_results = 'Mixed_Bagging_folds_' + str(dataset_name) + '.csv'
                    results_df.to_csv(nombre_results, encoding='utf_8_sig', index=False)

                    fold = fold + 1
                ''' End of fold for evaluation of test set '''

            run = run + 1
            ''' End of one run of k-fold evaluation '''


            ## Aggregation of results
            df_aggre_mean = results_df.groupby(['dataset', 'n_trees', 'model', 'perf_measure'], as_index=False)[
                'perf_value'].mean()
            df_aggre_mean.columns = ['dataset', 'n_trees', 'model', 'perf_measure', 'perf_value_mean']
            df_aggre_std = results_df.groupby(['dataset', 'n_trees', 'model', 'perf_measure'], as_index=False)[
                'perf_value'].std()
            df_aggre_std.columns = ['dataset', 'n_trees', 'model', 'perf_measure', 'perf_value_std']
            df_aggre_conf_matrix = \
                results_df.groupby(['dataset', 'n_trees', 'model', 'perf_measure'], as_index=False)[
                    'Conf_Matrix'].sum()

            df_aggre = pd.concat([df_aggre_mean, df_aggre_std.iloc[:, -1:]], axis=1)
            df_aggre = pd.concat([df_aggre, df_aggre_conf_matrix.iloc[:, -1:]], axis=1)

            # To save the results
            os.chdir(path_to_save)
            nombre_results_aggre = 'Mixed_Bagging_aggregated_' + str(dataset_name) + '.csv'
            df_aggre.to_csv(nombre_results_aggre, encoding='utf_8_sig', index=False)


            datasetcount = datasetcount + 1

    return


path_csv = os.chdir(root_path+'/Datasets/')
# Extraemos los nombres de todos los ficheros
total_name_list = []
for filename in os.listdir(path_csv):
    if filename.endswith('.csv'):
        total_name_list.append(filename)



# total_name_list = [  # 'teaching_assistant_MH.csv','chronic_kidney.csv','contraceptive_NL.csv',
    # 'balance_scale_BR.csv',
    #     'seismic-bumps.csv',
    #     'voting_records.csv',
    #     'phishing.csv',
    # 'monks_prob_3.csv','breast-w.csv','credit-a.csv',
    #     'contraceptive_LS.csv','wine_c1c3.csv',
    # 'tic-tac-toe.csv',
    #     'mammographic_mass.csv',
    #     'contraceptive_NS.csv','wine_c1c2.csv',
    # 'haberman.csv','cardiotocography_c1c3.csv','hepatitis.csv','titanic.csv',
    # 'internet_ad_cfs.csv','indian_liver_patients.csv',
    #     'monks_prob_2.csv',
    # 'breast_cancer.csv','arrhythmia_cfs.csv','vertebral_column.csv','sonar.csv',
    # 'spect_heart.csv',
    #     'horse_colic.csv',
    #     'cardiotocography_c2c3.csv',
    # 'monks_prob_1.csv',
    #     'credit-g.csv',
    #     'car_evaluation.csv',
    #     'climate_model.csv',
    # 'diabetic_retinopathy.csv','teaching_assistant_LM.csv','ionosphere.csv',
    # 'kr-vs-kp.csv',
    #     'pima_indians_diabetes.csv','cervical_cancer_risk_factors.csv',
    # 'cardiotocography_c1c2.csv',
    #     'teaching_assistant_LH.csv',
    # 'heart_disease.csv','balance_scale_LR.csv',
    # 'balance_scale_BL.csv','banknote_authentication.csv',
    # 'wine_c2c3.csv']


N= mp.cpu_count()

with mp.Pool(processes = N-20) as p:
        p.map(AdaptedMixedBagging, [dataset_name for dataset_name in total_name_list])
        # p.close()


