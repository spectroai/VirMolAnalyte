#!/usr/bin/env python
# coding: utf-8

import os
#把化学位移文件预测的路径加入到系统路径，方便导入
import sys
sys.path.append(r'.\VirMolAnalyte\VirDBcreator\NMRprediction')
import numpy as np
import torch
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from dataset import GraphDataset
from util import collate_reaction_graphs
# from model import nmrMPNN, training, inference
from sklearn.metrics import mean_absolute_error
from GetFeaturesOne import *
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
try:
    from rdkit.Chem.Draw import IPythonConsole
except Exception:
    IPythonConsole = None
if IPythonConsole is not None:
    IPythonConsole.ipython_useSVG = True
import DATAanalyte
from DATAanalyte.VectorSmilarity import *
from tqdm import tqdm

#绘制每个分子的碳原子编号
def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

#根据输入的smiles列表预测化学位移
def ShiftPrediction(smileslist):
    abnormal=[]
    shift_lists=[]
    for i,smi in enumerate(tqdm(smileslist)):
        try:
            smi=[smi]
            FeatureForOne=GetOneFeature(smi)
            train_y_mean = 95.86915330336912
            train_y_std = 51.61745076037435
            data_split = [0.0, 0.0, 1]
            batch_size = 128
            model_path = './VirMolAnalyte/VirDBcreator/NMRprediction/model/nmr_model.pt'
            random_seed = 1
            if not os.path.exists('./model/'): os.makedirs('./model/')
            #导入要预测的化合物的输入数据
            data = GraphDataset(FeatureForOne)
            train_set, val_set, test_set = split_dataset(data, data_split, shuffle=True, random_state=random_seed)
            test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)
            #获取节点特征和边的特征的维度
            node_dim = data.node_attr.shape[1]
            edge_dim = data.edge_attr.shape[1]
            #调用训练好的神经网络
            net = nmrMPNN(node_dim, edge_dim)
            net.load_state_dict(torch.load(model_path,map_location=torch.device("cpu")))
            # 使用模型进行预测
            Pred_values= inference(net, test_loader, train_y_mean, train_y_std)
            shift_lists.append(Pred_values)
        except:
            abnormal.append(i)
            shift_lists.append(None)
    return(shift_lists,abnormal)

#根据输入的smiles列表预测分子指纹
def FingerPrintGeneration(smileslist):
    MolecularFPall=[]
    AbnormalIndex=[]
    for i,smi in enumerate(smileslist):
        try:
            mol= Chem.MolFromSmiles(smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fp=fp.ToBitString()
            MolecularFPall.append(fp)
        except:
            AbnormalIndex.append(i)
            MolecularFPall.append(None)
            continue
    return(MolecularFPall,AbnormalIndex)

#除去单个化学位移小于10的分子，以及分子指纹中的空值
def datawashing(SmilesList,ShiftsList,FingerPrintList):
    SmilesList1=[]
    ShiftsList1=[]
    FingerPrintList1=[]
    for i in range(len(SmilesList)):
        if len(ShiftsList[i])>=10 and FingerPrintList[i]!=None:
            SmilesList1.append(SmilesList[i])
            ShiftsList1.append(ShiftsList[i])
            FingerPrintList1.append(FingerPrintList[i])
    return(SmilesList1,ShiftsList1,FingerPrintList1)

#批量获取碳类型以及在结构中的编号
def GetCarbonType(SmilesList):
    TypeListAll=[]
    NumberListAll=[]
    abnormal=[]
    for i,smi in enumerate(tqdm(SmilesList)):
        TypeList=[]
        NumberList=[]

        try:
            mol=Chem.MolFromSmiles(smi)
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 6:# 判断原子是否为碳原子，原子编号 6 代表碳
                    NumberList.append(atom.GetIdx())
                    num_hydrogens = atom.GetTotalNumHs()
                    if num_hydrogens == 3:
                        TypeList.append("q")
                    elif num_hydrogens == 2:
                        TypeList.append("t")
                    elif num_hydrogens == 1:
                        TypeList.append("d")
                    else:
                        TypeList.append("s")
            TypeListAll.append(np.array(TypeList))
            NumberListAll.append(np.array(NumberList))
        except:
            TypeListAll.append(None)
            NumberListAll.append(np.array(None))
            abnormal.append(i)
        if (i+1) % 10000 == 0:
            print('%d/%d processed' %(i+1, len(SmilesList)))
    return(TypeListAll,NumberListAll,abnormal)

#根据化学位移和碳类型生成向量
def batchCtypeVector(ShiftsList,CtypeList):
    VectorAll=[]
    abnormal=[]
    for i in tqdm(range(len(ShiftsList))):
        try:
            task=NMR2FringerPrint()
            vector=task.CTypeVectorGeneration(ShiftsList[i],CtypeList[i])
            #统计向量一共有多少个非零元素
            arr = np.array(vector)
            count = np.count_nonzero(arr)
            vector.append(count)
            VectorAll.append(vector)
        except:
            abnormal.append(i)
            VectorAll.append("CCC")
    return(VectorAll,abnormal)

#根据化学位移生成向量
def batchVector(ShiftsList):
    VectorAll=[]
    abnormal=[]
    for i in tqdm(range(len(ShiftsList))):
        try:
            task=NMR2FringerPrint()
            vector=task.NmrFingerprint(ShiftsList[i])
            #统计向量一共有多少个非零元素
            arr = np.array(vector)
            count = np.count_nonzero(arr)
            vector.append(count)
            VectorAll.append(np.array(vector))
        except:
            abnormal.append(i)
            VectorAll.append(None)
    return(VectorAll,abnormal)

#把化学位移数据后保留两位小数
def ShiftDataProcess(ShiftsList):
    ShiftsAll=[]
    for i in range(len(ShiftsList)):
        shifts=[round(shift,1) for shift in ShiftsList[i]]
        ShiftsAll.append(np.array(shifts))
        if (i+1) % 1000 == 0:
            print('%d/%d processed' %(i+1, len(ShiftsList)))
    return(ShiftsAll)


def VirDBfromSCIfinder(InputSdfFile,num=8):
    supplier = Chem.SDMolSupplier(InputSdfFile)
    smileslist=[]
    for mol in supplier:
        try:
            smiles=Chem.MolToSmiles(mol)
            smileslist.append(smiles)
        except:
            smileslist.append(None)
    CAS=[]
    Name=[]
    for mol in supplier:
        # 获取分子的属性信息
        try:
            props = mol.GetPropsAsDict()
            # 保存分子的属性信息
            for prop, value in props.items():
                if prop=="cas.rn":
                    CAS.append(value)
                elif prop=="cas.index.name":
                    Name.append(value)
        except:
            CAS.append(None)
            Name.append(None)


    print("chemical shift calculation")
    ShiftsList,abnormal_shifts_index=ShiftPrediction(smileslist)
    print(" CarbonType and CarbonNum generation")
    CarbonType,CarbonNum,abnormal_GetCarbonType=GetCarbonType(smileslist)
    print("CtypeNMRvector generation")
    CtypeNMRvector,abnormal_CtypeNMRvector=batchCtypeVector(ShiftsList,CarbonType)
    print("NMRvector generation")
    NMRvector,abnormal_NMRvector=batchVector(ShiftsList)
    abnormal=[]
    abnormal=abnormal_shifts_index+abnormal_GetCarbonType+abnormal_CtypeNMRvector+abnormal_NMRvector
    print("--------abnormal index--------")
    print(abnormal)
    dic={}
    dic["CAS"]=CAS
    dic["Name"]=Name
    dic["smiles"]=smileslist
    dic["shifts"]=ShiftsList
    dic["Ctype"]=CarbonType
    dic["CarbonNum"]=CarbonNum
    dic["CtypeNMRvector"]=CtypeNMRvector
    dic["FingerPrint"]=NMRvector        
    

    #基于scifinder下载的数据可能在生成数据库时会出现错误，因此需要去除这些错误
    CAS=[]
    Name=[]
    smiles=[]
    shifts=[]
    Ctype=[]
    CarbonNum=[]
    CtypeNMRvector=[]
    FingerPrint=[]
    for i in range(len(dic["shifts"])):
        if i not in abnormal and len(dic["shifts"][i])>=num:
            CAS.append(dic["CAS"][i])
            Name.append(dic["Name"][i])
            smiles.append(dic["smiles"][i])
            shifts.append(dic["shifts"][i])
            Ctype.append(dic["Ctype"][i])
            CarbonNum.append(dic["CarbonNum"][i])
            CtypeNMRvector.append(dic["CtypeNMRvector"][i])
            FingerPrint.append(dic["FingerPrint"][i])
    dic={}
    dic["CAS"]=CAS
    dic["Name"]=Name
    dic["smiles"]=smiles
    dic["shifts"]=shifts
    dic["Ctype"]=Ctype
    dic["CarbonNum"]=CarbonNum
    dic["CtypeNMRvector"]=CtypeNMRvector
    dic["FingerPrint"]=FingerPrint 
    return(dic,abnormal)

class VirDBgeneration(object):
    def VirDBgeneration(self,SmilesList,SavePath):
        self.SmilesList=SmilesList
        #化学位移预测
        ShiftsList,abnormal_shifts_index=ShiftPrediction(self.SmilesList)
        print(abnormal_shifts_index)
        #分子指纹预测
#         FingerPrints,AbnormalIndex=FingerPrintGeneration(self.SmilesList)
#         print(AbnormalIndex)
#         #数据清洗：除去单个化学位移小于10的分子，以及分子指纹中的空值
#         SmilesList,ShiftsList,FingerPrints=datawashing(self.SmilesList,ShiftsList,FingerPrints)
        #获取每个碳的类型以及编号
        CarbonType,CarbonNum,abnormal_GetCarbonType=GetCarbonType(self.SmilesList)
        #获取化合物的向量
        CtypeNMRvector,abnormal_CtypeNMRvector=batchCtypeVector(ShiftsList,CarbonType)
        NMRvector,abnormal_NMRvector=batchVector(ShiftsList)
        abnormal=[]
        abnormal=abnormal_shifts_index+abnormal_GetCarbonType+abnormal_CtypeNMRvector+abnormal_NMRvector
        print("--------abnormal index--------")
        print(abnormal)
        #以字典的格式保存数据
        dic={}
        dic["SmilesList"]=self.SmilesList
        dic["ShiftsList"]=ShiftsList
        dic["CarbonType"]=CarbonType
        dic["CarbonNum"]=CarbonNum
#         dic["FingerPrints"]=FingerPrints
        dic["CtypeNMRvector"]=CtypeNMRvector
        dic["NMRvector"]=NMRvector
        
        #去除异常
        smiles=[]
        shifts=[]
        Ctype=[]
        CarbonNum=[]
        CtypeNMRvector=[]
        FingerPrint=[]
        for i in range(len(dic["SmilesList"])):
            if i not in abnormal:
                smiles.append(dic["SmilesList"][i])
                
                shifts.append(dic["ShiftsList"][i])
                Ctype.append(dic["CarbonType"][i])
                CarbonNum.append(dic["CarbonNum"][i])
                CtypeNMRvector.append(dic["CtypeNMRvector"][i])
                FingerPrint.append(dic["NMRvector"][i])
        dic={}
        dic["smiles"]=smiles
        dic["shifts"]=shifts
        dic["Ctype"]=Ctype
        dic["CarbonNum"]=CarbonNum
        dic["CtypeNMRvector"]=CtypeNMRvector
        dic["FingerPrint"]=FingerPrint 
        self.dic=dic
    def VirDBfromSCIfinderFolder(self,FolderPath,SavePath,pysimplegui=False):
        CAS=[]
        Name=[]
        smiles=[]
        shifts=[]
        Ctype=[]
        CarbonNum=[]
        CtypeNMRvector=[]
        FingerPrint=[]
        list1=os.listdir(FolderPath)
        a=len(list1)
        for i in range(len(list1)):
            print("Total file %d，calculating %d"%(a, i))
            FolderPathOne=FolderPath+"/"+list1[i]
            dic,abnormal=VirDBfromSCIfinder(FolderPathOne)
            CAS+=dic["CAS"]
            Name+=dic["Name"]
            smiles+=dic["smiles"]
            shifts+=dic["shifts"]
            Ctype+=dic["Ctype"]
            CarbonNum+=dic["CarbonNum"]
            CtypeNMRvector+=dic["CtypeNMRvector"]
            FingerPrint+=dic["FingerPrint"]
            total=len(dic["CAS"])
            abnormal_num=len(set(abnormal))
            if pysimplegui==True:
                sg.one_line_progress_meter(
                list1[i], i, a,
                'Character Counter'
                )
            print("Finish file %d, total jobs %d, abnormal jobs %d"%(i,total,abnormal_num))
            print("__________________________________________________________________________________________")
        dic={}
        dic["CAS"]=CAS
        dic["Name"]=Name
        dic["smiles"]=smiles
        dic["shifts"]=shifts
        dic["Ctype"]=Ctype
        dic["CarbonNum"]=CarbonNum
        dic["CtypeNMRvector"]=CtypeNMRvector
        dic["FingerPrint"]=FingerPrint
        print("Finish all the jobs,a total of %d sets data have been generated."%(len(CAS)))
        self.dic=dic
        np.savez(SavePath,data=dic)







if __name__=="__main__":
    filename = r'D:\DATAanalyte2.0\database\Substance1.sdf'
    SavePath = r'D:\DATAanalyte2.0\database\coffeevirdb.npz'
    task=VirDBgeneration()
    task.VirDBfromSCIfinder(InputSdfFile=filename,SavePath= SavePath)





