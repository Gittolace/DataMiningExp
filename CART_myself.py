import pandas as pd

import operator

import loadData



#统计某种划分下不同label数目，生成字典
def labelCount(data):
    counts={}
    labels=data.iloc[:,-1]
    for one in labels:
        if one not in counts.keys():
            counts[one]=0
        counts[one]+=1
    return counts


#计算当前划分gini值
def computeGini(data):
    gini=1
    cnt=labelCount(data)
    for i in cnt.keys():
        p=float(cnt[i])/len(data)
        gini=gini-p**2

    return gini


#求当前划分最多的label对应key值
def maxLabelKey(data):
    cnt=labelCount(data)
    max=0
    maxKey=0
    for key in cnt:
        if(cnt[key]>max):
            max=cnt[key]
            maxKey=key
    return maxKey


#根据data中第i列数据与value关系划分data
def splitData(data,i,value,flag):
    if flag==0:
        subData=data[data.iloc[:,i]>value]
    else:
        subData=data[data.iloc[:,i]<=value]

    newData=subData.drop(subData.columns[i],axis=1)

    return newData

def chooseBestFeature(data):

    splitDic={}

    bestGini=99999.9

    bestFeature=0

    #对每一列进行每一个值分割，计算对应gini值，选出最小gini值
    for i in range(len(data.iloc[0,:])-1):
        valueList=set(data.iloc[:,i])
        bestColumnSplitGini=1000.0
        bestColumnSplitValue=0


        #每个列中每个值划分，计算最佳gini
        for value in valueList:
            newGini=0.0

            #根据value将data划分成大小两部分,计算分割后gini
            greaterData=splitData(data,i,value,0)
            p0=float(len(greaterData))/len(data)

            smallerData=splitData(data,i,value,1)
            p1=float(len(smallerData))/len(data)

            newGini=newGini+p0*computeGini(greaterData)+p1*computeGini(smallerData)

            #若newGini小于此列最好分割gini，则此分割为此列当前最好分割
            if newGini<bestColumnSplitGini:
                bestColumnSplitGini=newGini
                bestColumnSplitValue=value


        splitDic[data.columns[i]]=bestColumnSplitValue

        #若此列最好分割对应gini小于最好分割gini，则此分割为当前最好分割
        if bestColumnSplitGini<bestGini:
            bestGini=bestColumnSplitGini
            bestFeature=i


    bestFeatureValue=splitDic[data.columns[bestFeature]]

    return bestFeature,bestFeatureValue



#创建分类树函数，递归调用创建分类树
def createTree(data):

    #若数据集只剩最后一列，则返回此列数据集中最多Label对应key值
    if(len(data.columns)==1):
        return maxLabelKey(data)

    #若剩余label只有一种，则返回此label
    if(len(set(data.iloc[:,-1]))==1):
        return data.iloc[0,-1]

    bestFeat,bestFeatValue=chooseBestFeature(data)
    bestFeatLabel=data.columns[bestFeat]


    myTree={bestFeatLabel:{}}
    greaterData=splitData(data,bestFeat,bestFeatValue,0)
    myTree[bestFeatLabel]['>'+str(bestFeatValue)]=createTree(greaterData)
    smallerData=splitData(data,bestFeat,bestFeatValue,1)
    myTree[bestFeatLabel]['<='+str(bestFeatValue)]=createTree(smallerData)

    return myTree



data=loadData.load_data_csv()

print(createTree(data))
