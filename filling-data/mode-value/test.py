table = [[1,'?'], [1,2], [1,10], [1,5], [1,5],[1,'?'], [2,2]]
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
    print temp
    for r, row in enumerate(table):
        if table[r][i]=='?':
            table[r][i]=temp
    print table


mode(table, 1)

