

import pandas as pd

import numpy as np

import random

    

def load_value():
    f=open('StoneFlakes.dat',encoding='utf-8')
    dataList = []
    cnt=0
    for line in f:
        if cnt==0:
            line=line.replace("\n","")
            line=line.split()
            dataList.append(line)
            cnt=cnt+1
        else:
            line=line[:4]+','+line[5:]
            line=line.replace(" ", "")
            line=line.replace("\n","")

            s = line.split(',')
            dataList.append(s)
    f.close()

    return dataList

def load_label():
    f=open('Group.dat',encoding='utf-8')
    dataList = []
    for line in f:
        line=line.replace("\n","")
        line=line.split()
        dataList.append(line)

    label=[]
    for i in dataList[1:]:
        label.append(i[1])
    
    
    return label



def data_to_csv():
    value=load_value()
 
    label=load_label()

    #去掉第一行
    v=value[1:]

    for i in range(len(v)):
        #标签添加到最后一列
        v[i].append(label[i])
        #去掉ID列
        v[i]=v[i][1:]


    #去除数据集中未知属性值，用该属性平均值代替
    avgs=[]
    for j in range(len(v[0])):
        unKnownCnt=0
        sum=0.0;
        avg=0.0;
        for i in range(len(v)):
            if(v[i][j]=='?'):
                unKnownCnt=unKnownCnt+1
            else:
                sum=sum+float(v[i][j])
        avg=float(sum)/float(len(v)-unKnownCnt)
        avgs.append(avg)


    for j in range(len(v[0])):
        for i in range(len(v)):
            if(v[i][j]=='?'):
                #保留两位有效数字
                v[i][j]=format(avgs[j],'.2f')

    flag=[True,True,True,True]
    l=[]

    #为保证测试集所有类别都有，四种类别都取一个放到数据集末尾
    for i in range(len(v)):
        if(v[i][-1]=='1' and flag[0]):
            l.append(i)
            flag[0]=False
        elif(v[i][-1]=='2' and flag[1]):
            l.append(i)
            flag[1]=False
        elif(v[i][-1]=='3' and flag[2]):
            l.append(i)
            flag[2]=False
        elif(v[i][-1]=='4' and flag[3]):
            l.append(i)
            flag[3]=False


    for i in range(4):
        temp=v[l[i]]
        v[l[i]]=v[-1-i]
        v[-1-i]=temp

    
    
    data=pd.DataFrame(v,columns=['LBI','RTI','WDI','FLA','PSF','FSF','ZDF1','PROZE','LABEL'])

    data.to_csv("data.csv",index=False,sep=',')


#读CSV数据文件
def load_data_csv():

    data=pd.read_csv("data.csv")

    return data


