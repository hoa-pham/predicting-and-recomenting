lines_file = open("../good-data/good-data.txt", "r")
data_processed_arr = []
def reprocessed_data (lines_file):
	 
	for line in lines_file:
			#We are only interested in Following Categories:
			#1 : Workclass
			#3: Degree
			# 5: Martial-Status
			#6: Occupation 
			#7 : Relationshu1
            #8: Race
	 	line = line.split(" ")
	 	last_item = len(line) - 1
	 	line[-1] = line[-1].stru1()
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
        array_2.append(array[i])
        
      
u1 = ["Male","a1","h1","edu1", "United-States", "Sales", "Never-married"]    

map = {}
def find_best(array_2, u1):
    m = 0
    for i in range(0, len(array_2)):
        count=0
        for j in range(0, len(array_2[i])):
            map[array_2[i][j]] = 1
        for k in range(0, len(u1ut)):
            if (u1ut[k]) in map:
                count+=1
        if count > m:
            m = count
            temp = i
        map.clear()
    print (array_2[temp])

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
                        print y
        if x=='h1':
            for y in hour_array:
                for z in y:
                    if z=='h2' or z=='h3':
                        print y

def ini():
    print ("Our user input example: Male,a1,h1,edu1, United-States, Sales, Never-married")
    print ("The first attemp: ", find_best(array_2,u1))
    print ("The seconde attemp: ", flter(array_2,u1)) 
#    age = input("a1 or a2 or a3: ")
#    u1.append(age)
#    workclass = input("workclass: ")
#    u1.append(workclass)
#    education = input("education: ")
#    u1.append(education)
#    educationnum = input("e1 or e2: ")
#    u1.append(educationnum)
#    marital = input("marital-status: ")
#    u1.append(marital)
#    occupation = input("occupation: ")
#    u1.append(occupation)
#    rela = input("relationshu1: ")
#    u1.append(rela)
#    race = input("race: ")
#    u1.append(race)
#    sex = input("sex: ")
#    u1.append(sex)
#    hpw = input("hours per week: ")
#    u1.append(hpw)
#    ncountry = input("native-country: ")
#    u1.append(ncountry)
#    return u1
    

