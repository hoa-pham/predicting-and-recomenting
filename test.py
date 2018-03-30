import sys
table = [[1,2,3,'?'],[2,3, 3,'?'], [1,2,2,3]]
def ave(table, i):
    r = 0
    sum = 0
    for row in table:
        if table[r][i] == '?':
            r+=1
            continue 
        sum+=table[r][i]
        r+=1
    r = 0
    average = sum/len(table)
    for row in table:
        if table[r][i] =='?':
            table[r][i] = average
        r+=1

for row in table:
    i = 0
    for col in row:
        if col == '?':
            ave(table,i)
            break
        i+=1
print table
