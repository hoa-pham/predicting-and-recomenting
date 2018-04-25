#array_2 = [['b', 'c', 'd'], ['a', 'b', 'e', 'k', 'c', 'z', 'y'], ['b', 'e','c']]
#iput = ['b', 'e', 'c']
#map = {}
#def find_best(array_2, iput):
#    m = 0
#    for i in range(0, len(array_2)):
#        count=0
#        for j in range(0, len(array_2[i])):
#            map[array_2[i][j]] = 1
#        for k in range(0, len(iput)):
#            if map.has_key(iput[k]):
#                count+=1
#        if count>=m:
#            m = count
#            temp = i
#        map.clear()
#    print array_2[temp]

edu_array=[]
hour_array=[]
map = {}
inp = ["Male","a1","h1","edu1", "United-States", "Sales", "Never-married"]    

for x in arr:
    for y in x:
        map[y]=None
        if y=='edu1' or y=='edu2':
            edu_array.append(x)
        if y=='h1' or y=='h2' or y=='h3':
            hour_array.append(x)
def filter(arr, inp):
    for x in inp:
        if x=='edu1':
            for y in edu_array:
                for z in y:
                    if z=='edu2':
                        print y
        if x=='h1':
            for y in hour_array:
                for z in y:
                    if z=='h2' or z=='h3':
                        print y
            

filter(arr, inp)
