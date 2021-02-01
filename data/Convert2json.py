

with open('Advising_valid.txt', 'r') as f:
    dic = {}
    for line in f:
        if line =='':
            continue
        temp = line.rstrip().split('\t')
        if temp[0] not in dic:
            dic['id']= temp[0]
