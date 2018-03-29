lines_file = open("data.txt", "r")
data_processed_arr = []
def reprocessed_data (lines_file):
	 
	for line in lines_file:
			#We are only interested in Following Categories:
			#1 : Workclass
			#3: Degree
			# 5: Martial-Status
			#6: Occupation 
			#7 : Relationship
	 	line = line.split(",")
	 	last_item = len(line) - 1
	 	line[-1] = line[-1].strip()
	 	if ">50K" == line[last_item]:
	 		arr = [1,3,5,6,7]
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
    print(frequent_items_dict)
    
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


data_processed_arr = reprocessed_data(lines_file)
frequent_items(data_processed_arr, 30)


from apyori import apriori

rules = apriori(transactions, min_support = 0.06, min_confidence = 0.85, min_lift = 500, max_length = None)
results = list(rules)

results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]) + '\nStatistic:\t' + str(results[i][2]))
    



"""
import pyfpgrowth
data_processed_arr = reprocessed_data(lines_file)
patterns = pyfpgrowth.find_frequent_patterns(transactions, 3)
fp_rules = pyfpgrowth.generate_association_rules(patterns, 0.75)
"""




