# -*- coding: utf-8 -*-


lines_file = open("data.txt", "r")
transactions = []
def reprocessed_data (lines_file):
	file = []
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
	 		transactions.append(items)
	return transactions


transactions  = reprocessed_data(lines_file)

"""
from apyori import apriori
rules = apriori(transactions, min_support = 0.3, min_confidence = 0.75, min_lift = 0.0, max_length = None)
results = list(rules)

results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]) + '\nStatistic:\t' + str(results[i][2]))
"""
print (transactions[0]) 