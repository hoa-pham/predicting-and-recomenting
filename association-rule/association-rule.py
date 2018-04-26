lines_file = open("../good-data/good-data.txt", "r")


"""

This Function will preprocess data by the variables selection and 
return it as an array 


"""
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




data_processed_arr = reprocessed_data(lines_file)   

"""
This function will call in order to run Association Rules
The rules will be then under the results array
"""
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
        if "White" not in array[i]:
            array_2.append(array[i])
        
iput = ["Black","Male","a1","h1","u2", "United-States"]    
map = {}
index =0 
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
    index = temp  
    print ("our recomendation: ", array_2[temp])



#find_best(array_2, iput)
"""
geting user input
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
    
#first attempt
    """
    This is our first attempt 
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
        if max_count >= points_similar(rules[i], inp):
            rules_arr.append(rules[i])
    print (rules_arr)
    return rules_arr

inp = ["Male","a1","h1","edu1", "United-States", "Sales", "Never-married"]    

count = find_best_rule (array_2, inp)

#second attempt
""" 
This is our second attempt

"""
edu_array=[]
hour_array=[]
map = {}

for x in array_2:
    for y in x:
        map[y]=None
        if y=='edu1' or y=='edu2':
            edu_array.append(x)
        if y=='h1' or y=='h2' or y=='h3':
            hour_array.append(x)
def filter(arr, inp):
    for x in inp:
        if x=='edu1':
            for y in edu_array:
                for z in y:
                    if z=='edu2':
                        print (y)
        if x=='h1':
            for y in hour_array:
                for z in y:
                    if z=='h2' or z=='h3':
                        print (y)
            

filter(array_2, inp)