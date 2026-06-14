#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import os

# 动态添加路径，兼容打包后的环境
def get_nmr_prediction_path():
    """获取NMR预测模块的路径，兼容打包后的环境"""
    try:
        # PyInstaller 打包后的临时目录
        base_path = sys._MEIPASS
    except Exception:
        # 开发环境
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, 'VirMolAnalyte', 'VirDBcreator', 'NMRprediction')

sys.path.append(get_nmr_prediction_path())
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from padelpy import from_smiles,padeldescriptor
from poprogress import simple_progress
from dataset import GraphDataset
from GetFeaturesOne import *
from dgl.data.utils import split_dataset
import torch
from torch.utils.data import DataLoader
from util import collate_reaction_graphs
from VirMolAnalyte.model import nmrMPNN, training, inference

def get_resource_path(relative_path):
    """获取资源文件的绝对路径，兼容打包后的环境"""
    try:
        # PyInstaller 打包后的临时目录
        base_path = sys._MEIPASS
    except Exception:
        # 开发环境
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def get_temp_path(filename):
    """获取临时文件路径，兼容打包后的环境"""
    try:
        # PyInstaller 打包后的环境
        if hasattr(sys, '_MEIPASS'):
            # 获取exe文件所在的目录，而不是临时解压目录
            if hasattr(sys, 'frozen') and sys.frozen:
                # 打包后的环境，使用exe文件所在目录
                base_path = os.path.dirname(sys.executable)
            else:
                base_path = os.getcwd()
        else:
            # 开发环境
            base_path = os.getcwd()
    except Exception:
        # 开发环境
        base_path = os.getcwd()
    
    return os.path.join(base_path, filename)

def CtypeNumMW(smileslist):
    resultlist=[]
    for i in range(len(smileslist)):
        CH3=0
        CH2=0
        CH=0
        C=0
        mol=Chem.MolFromSmiles(smileslist[i])
        molecular_weight = Descriptors.MolWt(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:# 判断原子是否为碳原子，原子编号 6 代表碳
                num_hydrogens = atom.GetTotalNumHs()
                if num_hydrogens == 3:
                    CH3+=1
                elif num_hydrogens == 2:
                    CH2+=1
                elif num_hydrogens == 1:
                    CH+=1
                else:
                    C+=1
                    
        result=[CH3,CH2,CH,C,round(molecular_weight,2)]
        resultlist.append(result)
        
    return(resultlist)

def ShiftPrediction(smileslist):
    smileslist_remove=[]
    shift_lists=[]
    for i,smi in enumerate(simple_progress(smileslist)):
        try:
            smi=[smi]
            FeatureForOne=GetOneFeature(smi)
            train_y_mean = 95.86915330336912
            train_y_std = 51.61745076037435
            data_split = [0.0, 0.0, 1]
            batch_size = 128
            # 使用动态路径解析
            model_path = get_resource_path('VirMolAnalyte/VirDBcreator/NMRprediction/model/nmr_model.pt')
            random_seed = 1
            if not os.path.exists(get_temp_path('model')): os.makedirs(get_temp_path('model'))
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
            smileslist_remove.append(smi)
        except:
            continue
        if (i+1) % 10 == 0:
            print('%d/%d processed' %(i+1, len(smileslist)))
            
    return(shift_lists,smileslist)

def GetCarbonType(SmilesList):
    TypeListAll=[]
    NumberListAll=[]
    for i,smi in enumerate(SmilesList):
        TypeList=[]
        NumberList=[]
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
        if (i+1) % 10000 == 0:
            print('%d/%d processed' %(i+1, len(SmilesList)))
    return(TypeListAll,NumberListAll)

def Smi2PubchemFP(smileslist):
    df1=pd.DataFrame(smileslist)
    # 使用临时路径，兼容打包后的环境
    smi_file = get_temp_path('virdb.smi')
    csv_file = get_temp_path('descriptors.csv')
    
    df1.to_csv(smi_file, sep='\t', index=False, header=False)
    padeldescriptor(mol_dir=smi_file, 
                    d_file=csv_file,
                    fingerprints=True,
                    removesalt=True,
                    standardizenitro=True,
                    threads=20,
                    log=True
                   )


def VirDBGenerator(smileslist, output_path=None):
    print("_____chemical shift calculation_____")
    shiftlists,Normal_smileslist=ShiftPrediction(smileslist)
    CarbonType,CarbonNum=GetCarbonType(Normal_smileslist)
    
    print("_____PubchemFP calculation_____")
    Smi2PubchemFP(Normal_smileslist)
    # 使用临时路径读取descriptors.csv
    csv_file = get_temp_path('descriptors.csv')
    df=pd.read_csv(csv_file)
    df=df.drop("Name",axis=1)
    df1=df.dropna()
    normal=df1.index
    PuChemFP=np.array(df1)
    
    Normal_smileslist=[Normal_smileslist[i] for i in normal]
    shiftlists=[shiftlists[i] for i in normal]
    CarbonTypelist=[CarbonType[i] for i in normal]
    
    Ctype_Num_MW=CtypeNumMW(Normal_smileslist)
    DBindex=[i for i in range(len(shiftlists))]
    
    df2={}
    df2["smiles"]=Normal_smileslist
    df2["Vir_shifts"]=shiftlists
    df2["Ctype"]=CarbonTypelist
    df2["PubchemFP"]=PuChemFP
    df2["DBindex"]=DBindex
    df2["CtypeNum_and_MW"]= Ctype_Num_MW
    
    # 使用提供的输出路径，如果没有提供则使用默认路径
    if output_path is None:
        output_path = get_temp_path("Database/generated_virDB.npz")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    np.savez(output_path, data=df2)
    
    print("Job finish!")

def DatabaseCombine(DB1_link,DB2_link,savelink):
    DB1=np.load(DB1_link,allow_pickle=True)["data"][()]
    DB2=np.load(DB2_link,allow_pickle=True)["data"][()]
    df={}
    df["smiles"]=DB1["smiles"]+DB2["smiles"]
    df["Vir_shifts"]=DB1["Vir_shifts"]+DB2["Vir_shifts"]
    df["Ctype"]=DB1["Ctype"]+DB2["Ctype"]
    df["PubchemFP"]=np.vstack((DB1["PubchemFP"],DB2["PubchemFP"]))
    df["CtypeNum_and_MW"]= DB1["CtypeNum_and_MW"]+DB2["CtypeNum_and_MW"]
    df["DBindex"]=[i for i in range(len(df["smiles"]))]
    np.savez(savelink,data=df)