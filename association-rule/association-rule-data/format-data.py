import urllib
from bs4 import BeautifulSoup
file = open("input-data.txt", "w+")
page = urllib.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
soup = BeautifulSoup(page,"lxml")
text = soup.get_text()

for line in text:
    file.write(line)
file.close()
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
    if int(table[i][12])<=20:
        table[i][12] = '$'
    elif int(table[i][12])>20 and int(table[i][12])<=40:
        table[i][12] = '$$'
    else:
        table[i][12] = '$$$'
    i+=1

#writting svm to a file
for line in table:
    str = ' '.join(line)
    f.write("%s\n"%str)


