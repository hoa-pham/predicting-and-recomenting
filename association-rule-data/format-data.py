import urllib
from bs4 import BeautifulSoup
file = open("input-data.txt", "w+")
page = urllib.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
soup = BeautifulSoup(page,"lxml")
text = soup.get_text()

for line in text:
    file.write(line)
file.close()
import sys
fh = open("input-data.txt", "r+")
f = open("formatted-data.txt", "w+")

map = {}
table = []

#appending data to a 2D array table
for line in fh:
    line = line.replace(" ", "").rstrip('\n').split(",")
    table.append(line) 
del table[-1]

i=0
for row in table:
#hours per week
    if int(table[i][12])<=20:
        table[i][12] = 'h1'
    elif int(table[i][12])>20 and int(table[i][12])>=40:
        table[i][12] = 'h2'
    else: 
        table[i][12] = 'h3'

#age
    if int(table[i][0])<=25:
        table[i][0] = 'a1'
    elif int(table[i][0])>25 and int(table[i][0])<=40:
        table[i][0] = 'a2'
    elif int(table[i][0])>40 and int(table[i][0])<=60:
        table[i][0] = 'a3'
    else: 
        table[i][0] = 'a4'

#education-num
    if int(table[i][4])<=12:
        table[i][4] = 'edu1'
    elif int(table[i][4])>12 and int(table[i][4])<=14:
        table[i][4] = 'edu2'
    else: 
        table[i][4] = 'edu3'

    i+=1

#writting svm to a file
for line in table:
    str = ' '.join(line)
    f.write("%s\n"%str)
sys.exit()


