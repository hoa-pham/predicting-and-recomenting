lines_file = open("../good-data/good-data.txt", "r")
data_processed_arr = []
def reprocessed_data (lines_file):
	 
	for line in lines_file:
			#We are only interested in Following Categories:
			#1 : Workclass
			#3: Degree
			# 5: Martial-Status
			#6: Occupation 
			#7 : Relationship
            #8: Race
	 	line = line.split(" ")
	 	last_item = len(line) - 1
	 	line[-1] = line[-1].strip()
	 	if ">50K" == line[last_item]:
	 		arr = [0,1,4,5,6,7,8,9,12,13]
	 		items = [line[i] for i in arr]
	 		data_processed_arr.append(items)
	return data_processed_arr


def frequent_items (transactions, min_support_percentage):
    frequent_items_dict = {} 
    for transaction in transactions:
		#Count every duplicate items and store in dict
        for line in transaction:
            if line not in frequent_items_dict:
                frequent_items_dict[line] = 1
            else:
                frequent_items_dict[line] += 1
    
    return frequent_items_dict
    
	# Calculate the threshold. 
	#min_support_count = (min_support_percentage / 100)* number_data_records
	#print ("Total records is " + str(number_data_records))
	#print ("The support count is " + str(min_support_count))
	#frequent_items_dict = dict ((key,value) for key, value in frequent_items_dict.items() if value >= min_support_count)
	#for line in frequent_items_dict:
		#print (line , frequent_items_dict[line])
	#Next step, we need to remove all infrequent items in the each transcation 
	#print ("This is debug mode to check whether the infrequent is removed. and sorted")
	#for transaction in transactions:
		#transaction = list(filter(lambda x: x in frequent_items_dict, transaction))
		#transaction = sorted(transaction, key=lambda x : frequent_items_dict[x], reverse=True)
		#print (transaction)
    
	# Adding it to the FP Growth

def percentage_greater_than_40_hours (transactions):
    count = 0
    for transaction in transactions:
        hours = int(transaction[5])
        if hours >= 40:
            count += 1
    return count/len(transactions)
    



data_processed_arr = reprocessed_data(lines_file)   
#number = percentage_greater_than_40_hours(data_processed_arr)
#print (number)
#frequent_items(data_processed_arr, 30)


from apyori import apriori

rules = apriori(data_processed_arr, min_support = 0.02, min_confidence = 0.95, min_lift = 300, min_length =3, max_length = None)
results = list(rules)

array = []
array_2 = []
results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]) + '\nStatistic:\t' + str(results[i][2]))
    array.append(list(results[i][0]))
for i in range(0, len(array)):
    if len(array[i])>=6:
        for j in range(0, len(array[i])):
            if array[i][j] == 'h2' or array[i][j] == 'e2':
                array_2.append(array[i])
        
      
iput = ["White","Male","a1","h1"]    
map = {}
def find_best(array_2, iput):
    m = 0
    for i in range(0, len(array_2)):
        count=0
        for j in range(0, len(array_2[i])):
            map[array_2[i][j]] = 1
        for k in range(0, len(iput)):
            if (iput[k]) in map:
                count+=1
        if count > m:
            m = count
            temp = i
        
        map.clear()
    print ("our recomendation: ", array_2[temp])



find_best(array_2, iput)
"""
age: continuous. 
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
fnlwgt: continuous. 
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
education-num: continuous. 
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. 
sex: Female, Male. 
capital-gain: continuous. 
capital-loss: continuous. 
hours-per-week: continuous. 
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
"""
ip = []
def ini():
    age = input("a1 or a2 or a3: ")
    ip.append(age)
    workclass = input("workclass: ")
    ip.append(workclass)
    education = input("education: ")
    ip.append(education)
    educationnum = input("e1 or e2: ")
    ip.append(educationnum)
    marital = input("marital-status: ")
    ip.append(marital)
    occupation = input("occupation: ")
    ip.append(occupation)
    rela = input("relationship: ")
    ip.append(rela)
    race = input("race: ")
    ip.append(race)
    sex = input("sex: ")
    ip.append(sex)
    hpw = input("hours per week: ")
    ip.append(hpw)
    ncountry = input("native-country: ")
    ip.append(ncountry)
    return ip
    
#ini()

"""
from sklearn import preprocessing
import pandas as pd 
from xgboost import XGBClassifier
from sklearn import metrics

def format_data_encoding(data_set):
    # We need to convert all the value of the data in the value of 0 to 1 
    for column in data_set.columns:
        le = preprocessing.LabelEncoder()
        data_set[column] = le.fit_transform(data_set[column])
    return data_set

dataset = pd.read_csv("../numpy_formatted.txt")

#accuracy = metrics.accuracy_score(y_test, xg_pred)
"""

"""
import sys
map = {}
def find_best_rule(input_list):
    max = 0
    for i in range(0, len(array_2)):
        count=0
        for j in range(0, len(array_2[i])): 
            map[array_2[i][j]]=1
        for k in range(0, len(input_list)):
            if input_list[k] in map:
                count+=1
        map.clear()
        if count>=max:
            max = count
            tem = i
            print (tem)
    print (array_2[-1])

l = ["Wife"]
find_best_rule(l)
"""

"""



inp = ["Female"]


print (points_similar(array_2[-1], inp) )

"""

"""
def points_similar(rule, inp):
    max_length_inp = len(inp)
    count = 0
    for i in range(0,max_length_inp):
        if inp[i] in rule:
            count += 1
    return count


def find_best_rule (rules, inp):
    max_count = 0 
    for i in range(0, len(rules)):
        current_count = points_similar(rules[i],inp)
        if current_count > max_count:
            max_count = current_count
    
    rules_arr = []
    for i in range (0, len(rules)):
        if max_count == points_similar(rules[i], inp):
            rules_arr.append(rules[i])
    print (rules_arr)


inp = ["Female"]

find_best_rule (array_2, inp)

"""


"""
import pyfpgrowth
data_processed_arr = reprocessed_data(lines_file)
patterns = pyfpgrowth.find_frequent_patterns(transactions, 3)
fp_rules = pyfpgrowth.generate_association_rules(patterns, 0.75)
"""
