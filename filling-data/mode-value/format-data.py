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

#find mode of attribute
def mode(table, i):
    map={}
    for r, row in enumerate(table):
        if table[r][i]=='?':
            continue
        if map.has_key(table[r][i]):
            map[table[r][i]]+=1
            continue
        map[table[r][i]]=1
    max=0
    temp=0
    for key in map:
        if map[key]>max:
            max=map[key]
            temp=key
    for r, row in enumerate(table):
        if table[r][i]=='?':
            table[r][i]=temp

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
        if col == "?":
	    j+=1
            continue
        if not col.isdigit():
            table[i][j] = map[col]
        else:
            table[i][j] = int(col)
        j+=1
    i+=1
 
#filling "?" with attribute average value
for row in table:
    i = 0
    for col in row:
	if col == '?':
	    mode(table, i)
	    break
	i+=1
	
#writting svm to a file
for row in table:
    i=0
    for data in row:
	if i>0:
	    f.write('%d:%s '%(i, data)),
	else:
	    f.write('%s '%data),
	i+=1
    f.writelines('\n')
f.close()
