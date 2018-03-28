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

map = {}
table = []
#append data to a 2D array table
for line in fh:
    line = line.replace(" ", "").rstrip('\n').split(",")
    temp = line[0]
    line[0] = line[-1] 
    del line[-1]
    line.append(temp) 
    table.append(line) 

#mapping attributes to number
c = 0
for col in range(len(table[0])):
    r = 0
    value = 0
    for row in range(len(table)):
        key = table[r][c] 
        if key == "?":
            map[key] = 0
        if key.isdigit():
            break
        if map.has_key(key):
            r+=1
            continue
        map[table[r][c]] = value
        value+=1
        r+=1
    c+=1
print map

