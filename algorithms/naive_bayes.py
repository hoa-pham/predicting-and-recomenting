import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
def format_data_encoding(data_set):
    # We need to convert all the value of the data in the value of 0 to 1 
    for column in data_set.columns:
        le = preprocessing.LabelEncoder()
        data_set[column] = le.fit_transform(data_set[column])
    return data_set
        

data_set = pd.read_csv("numpy_formatted.txt")
data_set = format_data_encoding(data_set)




# split the data 

data_train , data_test = train_test_split (data_set, test_size = 0.3, random_state = 10, shuffle = True)

# Get all the features except the income
train_features = data_train.drop("income", axis = 1)
train_targets = data_train["income"]

test_features = data_test.drop("income", axis=1)
test_targets = data_test["income"]





clf = GaussianNB()
clf.fit(train_features, train_targets)

target_predict = clf.predict(test_features)
accuracy = accuracy_score(target_predict, test_targets, normalize=True)

print (accuracy)






#train 
