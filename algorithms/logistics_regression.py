# -*- coding: utf-8 -*-


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import preprocessing

dat = pd.read_csv("numpy_formatted.txt")
def format_data_encoding(data_set):
    # We need to convert all the value of the data in the value of 0 to 1 
    for column in data_set.columns:
        le = preprocessing.LabelEncoder()
        data_set[column] = le.fit_transform(data_set[column])
    return data_set

#encoded_data_set = format_data_encoding(dataset)


income_dummies = pd.get_dummies(dat['income'], prefix='income', drop_first=True)
dat = pd.concat([dat, income_dummies], axis=1)
del dat['income']

#Native.country:

#Convert value from non United States country to 'Not United States'
dat.loc[(dat['nativecountry']!='United-States') & (dat['nativecountry'].notnull()), 
        'nativecountry']='Not United States'

country_dummies = pd.get_dummies(dat['nativecountry'], prefix='origin_country', 
                                 drop_first=True,
                                dummy_na=True)

dat = pd.concat([dat, country_dummies], axis=1)
del dat['nativecountry']

#Capital gain/loss: Binned into 0 if capital loss == 0, 1 otherwise. Same for capital gain

dat.loc[dat['capitalloss']>0, 'capitalloss'] = 1
dat.loc[dat['capitalgain']>0, 'capitalgain'] = 1

#Sex:

sex_dummies = pd.get_dummies(dat['sex'], prefix='gender', drop_first=True)
dat = pd.concat([dat, sex_dummies], axis=1)
del dat['sex']

#Race: Binned into White/Non-White:

race_dict = {'Black': 'non_White', 'Asian-Pac-Islander': 'non_White',
             'Other': 'non_White', 'Amer-Indian-Eskimo': 'non_White'}

race_dummies = pd.get_dummies(dat['race'].replace(race_dict.keys(), race_dict.values()),
                              prefix='race', drop_first=True)
dat = pd.concat([dat, race_dummies], axis=1)
del dat['race']

#Occupation: Armed Forces binned with Protective-Service bc only 9 in the military. 

occupy_dict = {'Armed-Forces': 'Protective-serv-Military', 'Protective-serv': 
               'Protective-serv-Military'}

occupy_dummies = pd.get_dummies(dat['occupation'].replace(occupy_dict.keys(),
                                                          occupy_dict.values()),
                                                          prefix='occupation', drop_first=True,
                                                          dummy_na=True)
dat = pd.concat([dat, occupy_dummies], axis=1)
del dat['occupation']

#Marital Status: married subgroups binned into one group 'married':

married_dict = {'Married-civ-spouse': 'Married', 'Married-spouse-absent': 'Married',
                'Married-AF-spouse': 'Married'}

marital_dummies = pd.get_dummies(dat['maritalstatus'].replace(married_dict.keys(),
                                                               married_dict.values()),
                                                               prefix='marital_status',
                                                               drop_first=True)
dat = pd.concat([dat, marital_dummies], axis=1)
del dat['maritalstatus']

#education: binned

education_dict = {'1st-4th': 'Grade-school', '5th-6th': 'Grade-school', '7th-8th': 
                  'Junior-high', '9th': 'HS-nongrad', '10th': 'HS-nongrad', 
                  '11th': 'HS-nongrad', '12th': 'HS-nongrad', 'Masters': 
                  'Graduate', 'Doctorate': 'Graduate', 'Preschool': 'Grade-school'}

educ_dummies = pd.get_dummies(dat.education.replace(education_dict.keys(), 
                                                    education_dict.values()),
                                                    prefix='education',
                                                    drop_first=True)
                              
dat = pd.concat([dat, educ_dummies], axis=1)
del dat['education']

#workclass:

#Those who have a workclass of 'never worked' or 'without pay' will be dropped as we want to
#focus our attention on wage earners:

dat.drop(dat.loc[(dat.workclass=='Without-pay') | (dat.workclass=='Never-worked'), :].index,
        inplace=True)

class_dict = {'Local-gov': 'Government', 'State-gov': 'Government', 'Federal-gov': 'Government',
              'Self-emp-not-inc': 'Self-employed', 'Self-emp-inc': 'Self-employed'}

class_dummies= pd.get_dummies(dat.workclass.replace(class_dict.keys(), class_dict.values()),
                              prefix='workclass', drop_first=True, dummy_na=True)

dat = pd.concat([dat, class_dummies], axis=1)
del dat['workclass']

#relationship: not sure what this variable is about but with just a few levels, I will 
#create a set of dummies for all of them

relate_dummies = pd.get_dummies(dat.relationship, prefix='relationship', drop_first=True)

dat = pd.concat([dat, relate_dummies], axis=1)
del dat['relationship']


#Age

#applying a log transformation on age to maintain its interpretability and 
#make variable's scale closer to values of indicator variables.

dat['age'] = np.log10(dat['age'])

#fnlwgt: not quite sure the purpose of this variable with this data but holding
#on to it at least until we build baseline. Going to transform it to log scale

dat['fnlwgt'] = np.log10(dat['fnlwgt'])

#hours worked will be binned. 35-40hrs will be 'full-time'; <35 will be part-time;
#>40 will be '40+hrs'

dat['hours.worked'] = np.nan
dat.loc[(dat['hoursperweek']>=35) | (dat['hoursperweek']<=40), 'hours.worked'] = 'Full_time'
dat.loc[dat['hoursperweek']<35, 'hours.worked'] = 'Part_time'
dat.loc[dat['hoursperweek']>40, 'hours.worked'] = '40+hrs'

hours_dummies = pd.get_dummies(dat['hours.worked'], prefix='WklyHrs', drop_first=True)

dat = pd.concat([dat, hours_dummies], axis=1)

del dat['hoursperweek']
del dat['hours.worked']

#Education num will be binned in 4 year increments

educ_dict = {1: '1-4', 2: '1-4', 3: '1-4', 4: '1-4', 5: '5-8', 6: '5-8',
             7: '5-8', 8: '5-8', 9: '9-12', 10: '9-12', 11: '9-12', 12: '9-12',
             13: '13-16', 14: '13-16', 15: '13-16', 16: '13-16'}

educ_num = pd.get_dummies(dat['educationnum'].replace(educ_dict.keys(), educ_dict.values()),
                          prefix='YrsEduc', drop_first=False)

dat = pd.concat([dat, educ_num], axis=1)

del dat['educationnum']





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dat.loc[:, dat.columns != 'income_ >50K'],
                                                    dat.loc[:, 'income_ >50K'], test_size=.33,
                                                    random_state=1234)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=10, scoring='accuracy')








from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
#names = ["age", "workclass", "educationum","maritalstatus", "occupation","relationship","race","hoursperweek"]
lr = LinearRegression()

features = dat.loc[:, dat.columns != 'income_ >50K']
target = dat.loc[:, 'income_ >50K']
names  = dat.columns.values.tolist();
#rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=3)
rfe.fit(features,target)
print ("Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))

