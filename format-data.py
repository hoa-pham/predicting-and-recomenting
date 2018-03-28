#import urllib
#from bs4 import BeautifulSoup
#file = open("input-data.txt", "w+")
#
#page = urllib.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
#
#soup = BeautifulSoup(page,"lxml")
#text = soup.get_text()
#
#for line in text:
#    file.write(line)
#file.close()
#
import sys
fh = open("input-data.txt", "r+")
f = open("formatted-data.txt", "w+")


map = {}
table = []

#appending data to a 2D array table
i = 0
for line in fh:
    line = line.replace(" ", "").rstrip('\n').split(",")
    temp = line[0]
    line[0] = line[-1] 
    del line[-1]
    line.append(temp) 
    table.append(line) 
    #if i == 10:
    #    break
    i+=1

def ave(table, i):
    r = 0
    sum = 0
    for r in table:
        sum+=table[r][i]
        r+=1
    return sum/len(table[0])

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

for i, row in enumerate(table):
    for j, col in enumerate(table[i]):
        if col == "?":
            continue
        if not col.isdigit():
            table[i][j] = map[col]
        else:
            table[i][j] = int(col)
    print row



