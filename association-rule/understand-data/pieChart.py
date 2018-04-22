# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import scipy.stats as stats
import seaborn as sns

import pylab as pl

dataset = open("numpy_formatted.txt")

data_set = pd.read_csv("numpy_formatted.txt")

 
dict_martial_status = {} #5

dict_occupation_status = {} #6

dict_relationship_status = {} #7

dict_race_status = {} #8

dict_sex_status = {} #9

dict_education_status = {} #3

dict_country_status = {}

list_age = [] #0

list_number_education = [] #4

list_number_hours = [] #12

list_capital_gain = []

list_capital_loss = []

list_country = []

i = 0 
for data in dataset:
    if i >= 1:
        data = data.split(", ")
        list_age.append(data[0])
        list_number_education.append(data[4])
        list_number_hours.append(data[12])
        list_capital_gain.append(data[10])
        list_capital_loss.append(data[11])
        if data[13] not in dict_country_status:
            dict_country_status[data[13]] = 1
        else:
            dict_country_status[data[13]] += 1
        
        if data[3] not in dict_education_status:
            dict_education_status[data[3]] = 1
        else:
            dict_education_status[data[3]] += 1
        if data[5] not in dict_martial_status:
            dict_martial_status[data[5]] = 1
        else:
            dict_martial_status[data[5]] += 1
        if data[6] not in dict_occupation_status:
            dict_occupation_status[data[6]] = 1
        else:
            dict_occupation_status[data[6]] += 1
        if data[7] not in dict_relationship_status:
            dict_relationship_status[data[7]] = 1
        else:
            dict_relationship_status[data[7]] += 1
        if data[8] not in dict_race_status:
            dict_race_status[data[8]] = 1
        else:
            dict_race_status[data[8]] += 1
        if data[9] not in dict_sex_status:
            dict_sex_status[data[9]] = 1
        else:
            dict_sex_status[data[9]] += 1
    i += 1
    
        
        
        
        
        
        
def chart_pie (dic, message):
    province_population = []
    activities = []
    for key in dic:
        province_population.append(dic[key])
        activities.append(key)
    plt.pie(province_population, labels=activities, startangle=90,autopct='%.1f%%')
    plt.title(message)
    plt.show()
"""
chart_pie(dict_education_status, "Chart Education ")
chart_pie(dict_sex_status, "Chart Gender")
chart_pie(dict_race_status,"Chart Race")
chart_pie(dict_relationship_status, "Chart Relationship")
chart_pie(dict_occupation_status, "Chart Occupation")
chart_pie(dict_martial_status, "Chart Martial")
"""
chart_pie(dict_country_status, "Country")


def chart_barh (dic, message):
    province_population = []
    activities = []
    for key in dic:
        province_population.append(dic[key])
        activities.append(key)
    
    plt.figure(figsize=(5,4))
    y_pos = np.arange(len(activities))
    plt.barh(y_pos, province_population,align='center',alpha=0.5)
    plt.yticks(y_pos,activities)
    plt.xlabel("Number")
    plt.title(message)
    plt.show()    

"""
chart_barh(dict_education_status, "Education Portion")
chart_barh(dict_race_status,"Race Portion")
chart_barh(dict_relationship_status, "Relationship Portion")
chart_barh(dict_occupation_status, "Occupation Portion")
chart_barh(dict_martial_status, "Martial Portion")
"""



def histogram_chart (ls, message):
    h = [int(string) for string in ls]
    data = sorted(h)
    fit = stats.norm.pdf(data, np.mean(data), np.std(data))  #this is a fitting indeed
    pl.plot(data,fit,'-o')
    pl.hist(data,normed=True)      #use this to draw histogram of your data
    pl.title(message)
    pl.show()                   #use may also need add this 
    
"""
histogram_chart(list_age, "Distribution of Age")
histogram_chart(list_number_education, "Distribution of Education number")
histogram_chart(list_number_hours, "Distribution of number of working hours")
histogram_chart(list_capital_gain, "Distribution of Capital Gain")
histogram_chart(list_capital_loss, "Distribution of Capital Loss")
"""


gs = gridspec.GridSpec(7,2)
plt.figure(figsize=(25,34))
            



ax13 = plt.subplot(gs[6,1])
sns.countplot(y=data_set['nativecountry'], hue=data_set ['income'], ax=ax13, palette=['red', 
                                                                            'green'])
ax13.spines['top'].set_visible(False)
ax13.spines['right'].set_visible(False)
ax13.spines['left'].set_visible(False)
ax13.set_ylabel(""), ax13.set_xlabel("")
ax13.legend(title="Income Group")
ax13.set_title("Counts of Native Country by Income Group", fontsize=16, fontweight='bold')





















