# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import preprocessing

dataset = pd.read_csv("numpy_formatted.txt")
def format_data_encoding(data_set):
    # We need to convert all the value of the data in the value of 0 to 1 
    for column in data_set.columns:
        le = preprocessing.LabelEncoder()
        data_set[column] = le.fit_transform(data_set[column])
    return data_set

encoded_data_set = format_data_encoding(dataset)

X = encoded_data_set.iloc[:, [0,1,4,5,6,7,8,12,2,3,10,11]].values
y = encoded_data_set.iloc[:,14].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)



from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start= X_set[:,0].min() - 1 , stop = X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start= X_set[:,0].min() - 1 , stop = X_set[:,0].max() + 1, step = 0.01))
        

plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red','green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i , j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j,1], c= ListedColormap(('red','green'))(i),label = j)

plt.title("Logistic Regression (Training Set)")
plt.xlabel("number of years in education")
plt.ylabel("Number of hours working ")
plt.legend()
plt.show()

"""




from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
#names = ["age", "workclass", "educationum","maritalstatus", "occupation","relationship","race","hoursperweek"]
lr = LinearRegression()

features = encoded_data_set.drop("income",axis = 1)
target = encoded_data_set.income
names  = encoded_data_set.columns.values.tolist();
#rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=3)
rfe.fit(features,target)
print ("Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))




from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
"""
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=9, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
"""
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(features,target)

#display the feature importance  
print(model.feature_importances_)
print('\n',features.columns.values)

#bar plot of feature importance
values = model.feature_importances_
pos = np.arange(14) + 0.02
plt.barh(pos,values,align = 'center')
plt.title('Feature importance plot')
plt.xlabel('feature importance ')
plt.ylabel('features')
plt.yticks(np.arange(14),('age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status','occupation' ,'relationship', 'race' ,'sex', 'capital-gain', 'capital.loss','hours.per.week', 'native.country'))
plt.grid(True)










