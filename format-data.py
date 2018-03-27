import urllib
from bs4 import BeautifulSoup
import re
file = open("input-data.txt", "w+")

page = urllib.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')

soup = BeautifulSoup(page,"lxml")
text = soup.get_text()

for line in text:
    file.write(line)
file.close()

fh = open("input-data.txt", "r+")

for line in fh:
    print line

