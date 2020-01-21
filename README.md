## Repo Name: MLHelpers

### Summary of code in repo

This repo contains helper functions for experimenting with machine learning models in jupyter notebooks.

1. Features
>> dummy_var.py

One hot encodes categorical variables.

>>RemoveOutliers.py

Class for identifying outliers in data, and removal of outliers.

>> FeatureSelector.py

Class for performing feature selection for machine learning models or data preprocessing.

Implements four different methods to identify features for removal:

* Identifies features with a missing percentage greater than a specified threshold
* Identifies features with a single unique value
* Identifies collinear variables with a correlation greater than a specifie correlation coefficient
* Identifies features with the weakest relationship with the target variable

2. Models
>> MLPipepline.py

   Class for evaluating machine learning model, and outputs the following:
 * Confusion matrix
 * Learning curves
 * Model evaluation metrics
 * Feature Importance
 * Precision Recall Curves
 
 ### How to use the code
 
 Clone repo and import classes.
