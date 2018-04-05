#import urllib
#from bs4 import BeautifulSoup
#file = open("input-data.txt", "w+")
#page = urllib.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
#soup = BeautifulSoup(page,"lxml")
#text = soup.get_text()
#
#for line in text:
#    file.write(line)
#file.close()

fh = open("input-data.txt", "r+")
f = open("formatted-data.txt", "w+")

map = {}
table = []
table2 = []

#appending data to a 2D array table
i=0
for line in fh:
    line = line.replace(" ", "").rstrip('\n').split(",")
    temp = line[0]
    line[0] = line[-1] 
    del line[-1]
    line.append(temp) 
    table.append(line) 
    i+=1
del table[-1]


#mapping attributes to number
c = 0
for col in range(len(table[0])):
    r = 0
    value = 0
    for row in range(len(table)):
        key = table[r][c] 
        if key.isdigit():
            break
        if key == "?":
            map[key] = 0 
        if map.has_key(key):
            r+=1
            continue
        map[table[r][c]] = value
        value+=1
        r+=1
    c+=1

#switching data to svm array
i = 0
for row in table:
    j = 0
    for col in table[i]:
        if col == '?':
            j+=1
            continue
        if not col.isdigit():
            table[i][j] = map[col]
        else:
            table[i][j] = int(col)
        j+=1
    i+=1
k=0
for row in table:
    for col in row:
        if col=='?':
            table[k]=[]
    k+=1
table2 = [x for x in table if x!=[]]
	
#writting svm to a file
for row in table2:
    i=0
    for data in row:
        if data == '?':
            continue
	if i>0:
	    f.write('%d:%s '%(i, data)),
	else:
	    f.write('%s '%data),
	i+=1
    f.writelines('\n')
f.close()
