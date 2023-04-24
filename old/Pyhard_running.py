#### How to run
# In terminal: pyhard run # to obtain all results, then metadata_full.csv contains all the complexity measures
# In terminal: pyhard run --no-isa # to obtain complexity measures results without model information,
# then metadata.csv and ih.csv are the files of interest


### DOUBTS:
## How to automatically execute the code for several datasets?
## How to select the path where the results are saved?


##### Guidelines for input dataset
## Please follow the recommendations below:
# Only csv files are accepted
# The dataset should be in the format (n_instances, n_features)
# It cannot contains NaNs or missing values
# Do not include any index column. Instances will be indexed in order, starting from 1
# The last column should contain the target variable (y). Otherwise, the name of the target column must be declared in the field target_col (config file)
# Categorical features should be handled previously


