# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# For preprocessing the data
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score

def format_data_encoding(data_set):
    # We need to convert all the value of the data in the value of 0 to 1 
    for column in data_set.columns:
        le = preprocessing.LabelEncoder()
        data_set[column] = le.fit_transform(data_set[column])
    return data_set
        

data_set = pd.read_csv("numpy_formatted.txt")
encoded_data = data_set.copy()
encoded_data = format_data_encoding(encoded_data)


#We will ignore the education.  
#As we can see there is a correlation of education and education number 
sns.heatmap(encoded_data.corr(), square=True,cmap="Greens")
plt.show()
data_set[["education","educationnum"]].head(15)
 
(data_set["education"].value_counts() / data_set.shape[0]).head()





