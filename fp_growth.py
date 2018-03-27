lines_file = open("data.txt", "r")

def reprocessed_data (lines_file):
	 file = []
	 for line in lines_file:
	 	line = line.split(",")
	 	if line[len(line) - 1] == " >50K ":
	 		print (line[len(line) - 1])
	 return file

def frequent_items (transactions, min_support_percentage):
	lst = {}
	number_data_records = 0 
	for transaction in transactions:
		transaction = transaction.split(",")
		#We are only interested in Following Categories:
		#1 : Workclass
		#3: Degree
		# 5: Martial-Status
		#6: Occupation 
		#7 : Relationship
		arr = [1, 3, 5, 6, 7]
		items = [transaction[i] for i in arr]
		#Count every duplicate items and store in dict
		for line in items:
			if line not in lst:
				lst[line] = 0
			else:
				lst[line] += 1
		number_data_records += 1
	# Calculate the threshold. 
	min_support_count = (min_support_percentage / 100)* number_data_records
	print ("The support count is " + str(min_support_count))

	lst = dict ((key,value) for key, value in lst.items() if value >= min_support_count)
	for line in lst:
		print (line , lst[line])

frequent_items(lines_file, 30)


