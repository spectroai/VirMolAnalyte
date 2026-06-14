#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import os
# os.chdir("D:\VirMolAnalyte")  # Commented out to avoid hardcoded path issues

class MultiLabelDNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiLabelDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x=self.dropout(x)
        x = self.relu(self.fc2(x))
        x=self.dropout(x)
        x = self.relu(self.fc3(x))
        x=self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)

def NMRDEPTFPGen(shifts,Ctype):
    zero_array = np.zeros((4,256))
    rounded_shifts = [round(num) for num in shifts]
    for i,item in enumerate(rounded_shifts):
        item=int(item)
        if Ctype[i]=="s":
            if zero_array[0,item]==1:
                zero_array[0,item+1]=1
            else:
                zero_array[0,item]=1
        if Ctype[i]=="d":
            if zero_array[1,item]==1:
                zero_array[1,item+1]=1
            else:
                zero_array[1,item]=1
        if Ctype[i]=="t":
            if zero_array[2,item]==1:
                zero_array[2,item+1]=1
            else:
                zero_array[2,item]=1
        if Ctype[i]=="q":
            if zero_array[3,item]==1:
                zero_array[3,item+1]=1
            else:
                zero_array[3,item]=1
    return(zero_array.flatten())

def ShiftCompareOne(VecResult,peaks,therohold=0.8,method="shifts"):
    explt_list1=peaks.iloc[:,0].tolist()
    explt_list=explt_list1[:]
    data=VecResult
    hit_explt_all=[]
    hit_lib_all=[]
    delta=[]
    scores=[]
    MSEscores=[]
    if method=="shifts":
        #只比较化学位移
        for i in range(len(data["Vir_shifts"])):
            cmp_lib=data["Vir_shifts"][i].tolist()
            hit_lib=[]
            hit_explt=[]
            score=[]
            MinDeltaSum=0
            LibCarbonNum=len(cmp_lib)
            for j in range(LibCarbonNum):
                #计算数据库中一个化学位移值和explt_list的误差
                delta_all=[abs(cmp_lib[j]-m) for m in explt_list[:]]
                min_delta=min(delta_all)
                MinDeltaSum+=np.square(min_delta)
                #如果最小误差小于阈值，判断为1个hit
                if min_delta<=therohold:
                    hit=explt_list[delta_all.index(min(delta_all))]
                    hit_lib.append(cmp_lib[j])
                    hit_explt.append(hit)
                    #这里好像有个bug，如果说循环没有结束但是explt_list里面的元素已经被删除完了，就可能报错
                    explt_list.remove(hit)
                    if len(explt_list)==0:
                        break   
            #计算单个化合物的匹配得分
            score=round(len(hit_lib)/LibCarbonNum*100,1)
            scores.append(score)
            #计算MSEscore值
            MSEscore=MinDeltaSum/LibCarbonNum
            MSEscore=(2*math.atan(1/MSEscore)/(math.pi))
            MSEscores.append(MSEscore*100)
            #第一次循环完成后，开始数据库中第二个化合物比较，重新读取数据
            explt_list=peaks[:].iloc[:,0].tolist()
        #将化学匹配得分添加到结果文件中，以便于后续统计分析
        return(scores,MSEscores)
    
def DBfilter(IndexList,database):
    items=list(database.keys())
    df={}
    for i in range(len(items)):
        df[items[i]]=[database[items[i]][j] for j in IndexList]
    return(df)


# In[3]:


class Filters_and_evaluators(object):
    def __init__(self, TestSampleshift,TestsampleCtype,Database):
        self.TestSampleshift=TestSampleshift
        self.TestsampleCtype=TestsampleCtype
        self.database=Database
        self.NMR2FP= torch.load(r"./NMR2FP/FPDNNmodel.pth", map_location=torch.device('cpu'))
    def CarbonNumFilter(self,bias=2):
        #首先获得满足条件的索引
        shifts_num=len(self.TestSampleshift)
        IndexList=[]
        for i in range(len(self.database['CtypeNum_and_MW'])):
            CarbonNumber=sum(self.database['CtypeNum_and_MW'][i][0:4])
            if abs(CarbonNumber-shifts_num)<=bias:
                IndexList.append(i)
        self.database=DBfilter(IndexList,self.database)
    def CarbonTypeNumFilter(self,CarbonTypeNum,bias=2):
        IndexList=[]
        for i in range(len(self.database['CtypeNum_and_MW'])):
            CH3_bias=abs(self.database['CtypeNum_and_MW'][i][0]-CarbonTypeNum[0])
            CH2_bias=abs(self.database['CtypeNum_and_MW'][i][1]-CarbonTypeNum[1])
            CH_bias=abs(self.database['CtypeNum_and_MW'][i][2]-CarbonTypeNum[2])
            C_bias=abs(self.database['CtypeNum_and_MW'][i][3]-CarbonTypeNum[3])
            if CH3_bias<=bias and CH2_bias<=bias and CH_bias<=bias and C_bias<=bias:
                IndexList.append(i)
        self.database=DBfilter(IndexList,self.database)
                
    def MWFilter(self,MWlist,bias=5):
        IndexList=[]
        for i in range(len(self.database['CtypeNum_and_MW'])):
            MW=self.database['CtypeNum_and_MW'][i][4]
            for mw in MWlist:
                if abs(MW-mw)<=bias:
                    IndexList.append(i)
                    break
        self.database=DBfilter(IndexList,self.database)
    def FPS_evaluator(self):
        #首先首先需要把13C DEPT NMR转化为1024的向量
        self.NMRvector=NMRDEPTFPGen(self.TestSampleshift,self.TestsampleCtype)
        X_test_tensor = torch.tensor(self.NMRvector, dtype=torch.float32)
        Y=self.NMR2FP(X_test_tensor)
        Y_detached=Y.detach()
        smiList=[]
        for i in range(len(self.database["PubchemFP"])):
            smi=cosine_similarity(np.array(Y_detached),np.array(self.database["PubchemFP"][i]))
            smiList.append(smi)
        self.FPSscore=smiList
    def CSS_AAS_evaluator(self):
        peaks=pd.DataFrame(self.TestSampleshift)
        self.CSSscore,self.AASscore=ShiftCompareOne(self.database,peaks)
    def FPAACS_evaluator(self,weights=(0.28, 0.16, 0.56),):
        self.FPS_evaluator()
        self.CSS_AAS_evaluator()
        self.FPAACSscore=np.array(self.AASscore)*weights[0]+ np.array(self.CSSscore)*weights[1]+np.array(self.FPSscore)*100*weights[2]
    def ShowTopN(self,scorelist,TopN=10):
        original_list = scorelist
        indexed_list = list(enumerate(original_list))
        # 根据值进行降序排序
        sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
        # 提取排序后的索引和排序后的值
        sorted_indices = [index for index, value in sorted_list][0:TopN]
        sorted_values = [value for index, value in sorted_list][0:TopN]
        df={}
        df["smiles"]=[self.database["smiles"][i] for i in sorted_indices]
        df["Vir_shifts"]=[self.database["Vir_shifts"][i] for i in sorted_indices]
        df["Ctype"]=[self.database["Ctype"][i] for i in sorted_indices]
        df["score"]=sorted_values
        df["DBindex"]=[self.database["DBindex"][i] for i in sorted_indices]
        self.TOPN=pd.DataFrame(df)





