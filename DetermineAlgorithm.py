import math

def distance(algorithm,name,index):
    dis=(algorithm[index[0]]-name[0])**2+(algorithm[index[1]]-name[1])**2+(algorithm[index[2]]-name[2])**2

    return dis



algorithm=[12,13,14,2,3,20,18,1,7,16]
name=[23,25,8]
minDistance=99999999
minIndex=[0,1,2]
dis=0

for i in range(8):
    for j in range(i+1,9):
        for k in range(j+1,10):
            dis=distance(algorithm,name,[i,j,k])
            if(dis<minDistance):
                minDistance=dis
                minIndex=[i,j,k]



print(algorithm[minIndex[0]],algorithm[minIndex[1]],algorithm[minIndex[2]])

#分类与回归树,随机森林,Gradient Tree Boosting
