###########################################################################################################
####  In this script we carry out an EDA of the datasets from the point of view of performance
#### and complexity in order to determine the characteristics of those datasets that benefit more
#### from the application of ensemble methos
###########################################################################################################

# We would like to answer the following questions:
# When to use ensemble methods?
# Do harder datasets benefit more from ensemble methods?
# Which datasets obtain better accuracy with ensemble rather than single classifiers?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


root_path = os.getcwd()

path_csv = os.chdir(root_path+'/Results_general_algorithm')
df_complexity = pd.read_csv('complex_info_total.csv')
path_csv = os.chdir(root_path+'/Classification_SingleLearner')
df_classif = pd.read_csv('ClassificationSingleLearner_AllDatasets.csv')

# We keep only the 9 selected complexity measures: Hostility, kDN, DCP, TD_U, CLD, N1, N2, LSC, F1
df_complexity.drop(['DS','TD_P','MV','CB','LSradius','H','U',
                    'F2','F3','F4'], axis=1, inplace=True)

# Join of two tables
df_classif = df_classif.join(df_complexity, on='Dataset')
df = pd.merge(df_complexity, df_classif, left_on='Dataset', right_on='dataset', how='left').drop('dataset', axis=1)

## HCER AHORA UN BUEN EDA
