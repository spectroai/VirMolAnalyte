#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
from pathlib import Path

# 获取当前 Notebook 的路径
notebook_path = Path.cwd()

# 设置工作目录为当前 Notebook 所在的目录
os.chdir(notebook_path)
import sys
sys.path.append(r'.\VirMolAnalyte\VirDBcreator\NMRprediction')
import VirMolAnalyte
# from VirMolAnalyte.Filter_evaluator import MultiLabelDNN
from VirMolAnalyte.Filter_evaluator import *
from VirMolAnalyte.NMR1D import *
import torch
import torch.nn as nn
import PySimpleGUI as sg
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from VirMolAnalyte.DataPrepare import *
import pandas as pd


# In[3]:


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
NMR2FP= torch.load(r"./NMR2FP/FPDNNmodel.pth", map_location=torch.device('cpu'))
# 将图像转换为可以在PySimpleGUI中显示的格式
def get_image_data(image_path, maxsize=(100, 100)):
    image = Image.open(image_path)
    image.thumbnail(maxsize)
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()


def plot_molecules_with_legend(smiles_list, legends,grid_width=4, image_size=(200, 200), output_filename="molecules_grid.png"):

    # 将 SMILES 转换为 RDKit 分子对象
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    # 生成每个分子的图像，并保存到列表中
    images = [Draw.MolToImage(mol, size=image_size, kekulize=True) for mol in molecules]

    # 计算所需的网格高度
    grid_height = (len(images) + grid_width - 1) // grid_width

    # 创建一个空白图像
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width * 2, grid_height * 2))

    # 将每个分子图像粘贴到组合图像中，并添加图例
    for index, (img, smiles,legend) in enumerate(zip(images, smiles_list,legends)):
        ax = axs[index // grid_width, index % grid_width]
        ax.imshow(img)
        ax.axis('off')  # 关闭坐标轴
        # 在图像下方添加图例
        ax.text(0.5, -0.1, legend, ha='center', va='top', fontsize=10, transform=ax.transAxes)

    # 处理多余的空白子图
    for i in range(len(images), grid_width * grid_height):
        axs[i // grid_width, i % grid_width].axis('off')

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.show()


# In[4]:


def VirMolAnalyte(shifts,Ctype,dabatase,filters,evaluator,TopN=10,
                  CNFbias=5,CTNFbias=2,MWlist=[300],MWbias=5):
    #数据库比较
    task=Filters_and_evaluators(np.array(shifts),np.array(Ctype),dabatase)

    #加上CarbonNumFilter
    if "CNF" in filters:
        task.CarbonNumFilter(bias=CNFbias)
    #加上CarbonTypeNumFilter
    if "CNF" in filters:
        task.CarbonTypeNumFilter(CarbonTypeNum=[1,2,3,4],bias=CTNFbias)
    #加上MWFilter
    if "MW" in filters:
        task.MWFilter(MW=MWlist,bias=MWbias)
        
    DBIndex=task.database["DBindex"]
    
    if evaluator=="FPS":
        task.FPS_evaluator()
        task.ShowTopN(task.FPSscore,TopN=TopN)
        result=task.TOPN
    if evaluator=="AAS" or evaluator=="CSS":
        task.CSS_AAS_evaluator()
        if evaluator=="AAS":
            task.ShowTopN(task.AASscore,TopN=TopN)
            result=task.TOPN
        if evaluator=="CSS":
            task.ShowTopN(task.CSSscore,TopN=TopN)
            result=task.TOPN
    if evaluator=="FPAACS":
        task.FPAACS_evaluator()
        task.ShowTopN(task.FPSscore,TopN=TopN)
        result=task.TOPN
    return(result)


# In[5]:


#-------------------------------------页面布局————————————————————————
# sg.theme('LightBlue')
sg.theme('DarkTeal9')
def layout11gen():
    layout11=[
             [sg.Text("STEP1: Enter the path of the NMR data:", font=("Helvetica", 12, 'bold'))],
            [sg.FolderBrowse(button_text="13C_NMR", button_color=('white', 'blue')), sg.Input("", size=(35, 1), font=("Helvetica", 12))],
            [sg.FolderBrowse(button_text="DEPT90", button_color=('white', 'blue')), sg.Input("", size=(35, 1), font=("Helvetica", 12))],
            [sg.FolderBrowse(button_text="DEPT135", button_color=('white', 'blue')), sg.Input("", size=(35, 1), font=("Helvetica", 12))],
              #step2
             [sg.Text("STEP2: Get peak", font=("Helvetica", 12, 'bold'))],
            [sg.Text("Threshold_C:", font=("Helvetica", 10)), sg.Input(3000000.0, size=(20, 1), key="-threshold_C-", font=("Helvetica", 11))],
            [sg.Text("Threshold_C90:", font=("Helvetica", 10)), sg.Input(90000000.0, size=(20, 1), key="-threshold_C90-", font=("Helvetica", 11))],
            [sg.Text("Threshold_C135pos:", font=("Helvetica", 10)), sg.Input(50000000.0, size=(20, 1), key="-threshold_C135pos-", font=("Helvetica", 11))],
            [sg.Text("Threshold_C135neg:", font=("Helvetica", 10)), sg.Input(-50000000.0, size=(20, 1), key="-threshold_C135neg-", font=("Helvetica", 11))],
            [sg.Button("Submit", key="-submit1-", button_color=('white', 'green')), sg.Button("Combine", key="-combine-", button_color=('white', 'orange'))],
              #step3
              [sg.Text("STEP3: Solvent removal:", font=("Helvetica", 12, 'bold'))],
            [sg.Text("Solvent:", font=("Helvetica", 10)), sg.Combo(["Chloroform", "Methanol","DMSO","Pyridine"], key="-solvent-", size=(15, 1)),
             sg.Text("Type:", font=("Helvetica", 10)), sg.Combo(["CH3", "CH2", "CH", "C"], size=(10, 1), key="-type-"),
             sg.Button("Submit", key="-submit2-", button_color=('white', 'green'))],
    
                [sg.Text("STEP4: Impurity removal:", font=("Helvetica", 12, 'bold'))],
                [sg.Text("Type:", font=("Helvetica", 10)), sg.Combo(["CH3", "CH2", "CH", "C"], size=(5, 1), key="-typeimpurity-"),
                 sg.Text("Threshold:", font=("Helvetica", 10)), sg.Input("", size=(5, 1), key="-thresholdimpurity1-", font=("Helvetica", 11)),
                 sg.Input("1e7", size=(5, 1), key="-thresholdimpurity2-", font=("Helvetica", 11)),
                 sg.Button("Submit", key="-submit3-", button_color=('white', 'green'))],
              #step5
        
                [sg.Text('STEP5:In silico analysis:',font=("Helvetica", 12, 'bold'))],
                [sg.Text('In silico database selection',font=("Helvetica", 10, 'bold'))],

                 [sg.Radio('All', group_id='-OPTIONS-',key='OPTION1',default=False), 
                 sg.Radio('Plant',group_id='-OPTIONS-', key='OPTION2',default=True), 
                 sg.Radio('Human',group_id='-OPTIONS-', key='OPTION3',default=False), 
                 sg.Radio('Microorganism', group_id='-OPTIONS-',key='OPTION4',default=False),
                 sg.Radio('Drug', group_id='-OPTIONS-',key='OPTION5',default=False)],
                [sg.FileBrowse(button_text="Other database", button_color=('white', 'gray')),
                sg.Input("", size=(35, 1))],
                [sg.Button('Load database',key="-submit3.5-"),sg.Button('Load other database', key="-submit3.6-", button_color=('white', 'gray'))],

                [sg.Text('Filter parameters',font=("Helvetica", 10, 'bold'))],
                [sg.Checkbox('CNF', key='OPTION6',default=True), 
                 sg.Checkbox('CTNF', key='OPTION7',default=True), 
                 sg.Checkbox('MW', key='OPTION8',default=False)], 
                [sg.T("CNF bias"), sg.In(5,size=(4,1),key="-CNFbias-"),
                 sg.T("CTNF bias"), sg.In(2,size=(4,1),key="-CTNFbias-"),
                 sg.T("MW list"), sg.In("300,400",size=(5,1),key="-MWlist-"),
                ],  

                [sg.Text('Evaluator parameters',font=("Helvetica", 10, 'bold'))],
                [sg.Radio('CSS',group_id='-OPTIONS1-', key='OPTION9'), 
                 sg.Radio('AAS', group_id='-OPTIONS1-',key='OPTION10'), 
                 sg.Radio('FPS', group_id='-OPTIONS1-',key='OPTION11'), 
                 sg.Radio('FPAACS',group_id='-OPTIONS1-', key='OPTION12',default=True)],

              [sg.T("CSS_thres"), sg.In(0.5,size=(4,1)),
                 sg.T("FPAACS_weight"), sg.In("0.2,0.3,0.5",size=(10,1),key="-weights-"),
                ],  
                [sg.Button('Submit Task',key="-submit4-")]
             ]
    return (layout11)


# In[20]:


smileslist=[]
scorelist=[]
DBindex=[]
layout12 = []
layout12.append([sg.T("Analysis Report by VirMolAnalyte",font=("Arial",16),background_color="white",text_color="black",pad=((300,10),(20,0)))])
layout12.append([sg.T("1.The original 13C DEPT NMR spectrum",font=("Arial",14),background_color="white",text_color="black",pad=((50,10),(20,0)))])
layout12.append([sg.Image(r".\GUI_start_files\fig1.png",background_color='white',key="-fig1-",pad=((0,10),(0,0)))])

layout12.append([sg.T("2.The combined 13C DEPT NMR spectrum",font=("Arial",14),background_color="white",text_color="black",pad=((50,10),(20,0)))])
layout12.append([sg.Image(r".\GUI_start_files\fig2.png",background_color='white',key="-fig2-",pad=((0,10),(0,0)))])
layout12.append([sg.T("3.The VirMolAnalyte analysis result",font=("Arial",14),background_color="white",text_color="black",pad=((50,10),(20,0)))])
layout12.append([sg.Image(r".\GUI_start_files\fig4.png",background_color='white',key="-fig4-",pad=((70,10),(0,0)))])
layout12.append([sg.T("Compound ID:"), sg.In(size=(5,1),key="-DBindex-"),sg.B("Submit",key=("-submit5-")),sg.T("shifts:"),
                 sg.Multiline(size=(20,4),key="-shifts-"),sg.T("Ctype:"),sg.Multiline(size=(20,4),key="-Ctype-"),
                 sg.T("smiles:"),sg.Multiline(size=(20,4),key="-smiles-")])
# layout12.append([sg.LB(smileslist,size=(40,20),key="-smiles-",pad=((10,0),(0,0))),sg.LB(scorelist,size=(25,20),key="-score-"),
#        sg.LB(DBindex,size=(25,20),key="-DBindex-"),sg.Image(r".\GUI_start_files\fig3.png",background_color='white',key="-fig1-")])

layout21 = []
layout21.append([sg.Image(r".\GUI_start_files\DBgen.png",key="-fig2-",pad=((0,10),(0,0)))])
layout21.append([sg.FileBrowse(button_text="smiles",target="-In21-"),sg.In("smiles.csv",size=(30,1),key="-In21-"),sg.B("Submit",key="-submit21-")])


def create_window():
    menu_def=[
        ['File','open file::openkey'],
        ['Edit','edit1'],
        ['Help','about']
         ]
    layout11=layout11gen()
     
    layout=[[sg.Menu(menu_def)],
         [sg.B("VirMolAnalyte",button_color="red",pad=((0,0),(0,0)),key="-func1-"),
          sg.B("DBcreator",button_color="#308E98",pad=((0,0),(0,0)),key="-func2-"),sg.T("All database 605,735 NPs; Plant 188,478 NPs; Human 217,347 NPs; Microorganism 36,427",text_color="red",background_color="blue")],
         [sg.Frame("",layout11,visible=True,size=(400,780),key="-element11-"),
          sg.Col(layout12,key="-element12-",size=(900,780), background_color='white',vertical_scroll_only=True,scrollable=True,visible=True),
          sg.Frame("",layout21,size=(1200,900),key="-element21-",visible=False),
         ]
         
        ]
    
    return sg.Window('VirMolanalyte', layout,keep_on_top=False,element_justification='left',resizable=True)

window= create_window()
while True:
    event,values=window.read()
    print(event)
    print(values)
    if event==None:
        break
    if event=="-submit1-":
        try:
            data_dir=values['13C_NMR']
            data_dir135=values['DEPT135']
            data_dir90=values["DEPT90"]
            test=CNMR_process(data_dir,data_dir135,data_dir90)
            test.get_13CDEPT_peak(threshold_C=float(values['-threshold_C-']),threshold_C90=float(values['-threshold_C90-']),
                                  threshold_C135pos=float(values['-threshold_C135pos-']),threshold_C135neg=float(values['-threshold_C135neg-']),DrawWin="True")
            window['-fig1-'].update(r"./GUI_result_files/fig1.png")
        except Exception as e:
             sg.popup(e)
    
    if event=="-combine-":
        try:
            test.sort_Ctype_mindelta()
            test.CNMR_reconstract(width=0.6,fontsize='medium',chemical_shift="False",label="False",summary="True",x_left=200,x_right=-2,y_top=0.6*1e8,y_bottom=-0.6*1e8,DrawWin="True")
            window['-fig2-'].update("./GUI_result_files/fig2.png")
            test.combine_data(CarbonFileName="NMR-1D.csv")
        except Exception as e:
             sg.popup(e)
    if event=="-submit2-":
        try:
            if values["-type-"]=="CH3":
                test.solvent_remove(test.hit_CH3,type="CH3", solvent=values["-solvent-"])
                test.CNMR_reconstract(width=0.6,fontsize='medium',chemical_shift="False",label="False",summary="True",x_left=200,x_right=-2,y_top=0.6*1e8,y_bottom=-0.6*1e8,DrawWin="True")
                window['-fig2-'].update("./GUI_result_files/fig2.png")
            if values["-type-"]=="CH2":
                test.solvent_remove(test.hit_CH2,type="CH2", solvent=values["-solvent-"]) 
                test.CNMR_reconstract(width=0.6,fontsize='medium',chemical_shift="False",label="False",summary="True",x_left=200,x_right=-2,y_top=0.6*1e8,y_bottom=-0.6*1e8,DrawWin="True")
                window['-fig2-'].update("./GUI_result_files/fig2.png")
            if values["-type-"]=="CH":
                test.solvent_remove(test.hit_CH,type="CH", solvent=values["-solvent-"])
                test.CNMR_reconstract(width=0.6,fontsize='medium',chemical_shift="False",label="False",summary="True",x_left=200,x_right=-2,y_top=0.6*1e8,y_bottom=-0.6*1e8,DrawWin="True")
                window['-fig2-'].update("./GUI_result_files/fig2.png")
            if values["-type-"]=="C":
                test.solvent_remove(test.Hit_C_or_unhited,type="C", solvent=values["-solvent-"])
                test.CNMR_reconstract(width=0.6,fontsize='medium',chemical_shift="False",label="False",summary="True",x_left=200,x_right=-2,y_top=0.6*1e8,y_bottom=-0.6*1e8,DrawWin="True")
                window['-fig2-'].update("./GUI_result_files/fig2.png")
            test.combine_data(CarbonFileName="NMR-1D.csv")
        except Exception as e:
             sg.popup(e)
    if event=="-submit3-":
        try:
            if values["-typeimpurity-"]=="CH3":
                test.impurity_removal(test.hit_CH3,type="CH3",rate=0.5,Standsrd_num=0,method="absolute",threshold=float(values["-thresholdimpurity1-"])*float(values["-thresholdimpurity2-"]))
                test.CNMR_reconstract(width=0.6,fontsize='medium',chemical_shift="False",label="False",summary="True",x_left=200,x_right=-2,y_top=0.6*1e8,y_bottom=-0.6*1e8)
                window['-fig2-'].update("./GUI_result_files/fig2.png")
            if values["-typeimpurity-"]=="CH2":
                test.impurity_removal(test.hit_CH2,type="CH2",rate=0.5,Standsrd_num=0,method="absolute",threshold=float(values["-thresholdimpurity1-"])*float(values["-thresholdimpurity2-"]))
                test.CNMR_reconstract(width=0.6,fontsize='medium',chemical_shift="False",label="False",summary="True",x_left=200,x_right=-2,y_top=0.6*1e8,y_bottom=-0.6*1e8)
                window['-fig2-'].update("./GUI_result_files/fig2.png")
            if values["-typeimpurity-"]=="CH":
                test.impurity_removal(test.hit_CH,type="CH",rate=0.5,Standsrd_num=0,method="absolute",threshold=float(values["-thresholdimpurity1-"])*float(values["-thresholdimpurity2-"]))
                test.CNMR_reconstract(width=0.6,fontsize='medium',chemical_shift="False",label="False",summary="True",x_left=200,x_right=-2,y_top=0.6*1e8,y_bottom=-0.6*1e8)
                window['-fig2-'].update("./GUI_result_files/fig2.png")
            if values["-typeimpurity-"]=="C":
                test.impurity_removal(test.Hit_C_or_unhited,type="C",rate=0.5,Standsrd_num=0,method="absolute",threshold=float(values["-thresholdimpurity1-"])*float(values["-thresholdimpurity2-"]))
                test.CNMR_reconstract(width=0.6,fontsize='medium',chemical_shift="False",label="False",summary="True",x_left=200,x_right=-2,y_top=0.6*1e8,y_bottom=-0.6*1e8)
                window['-fig2-'].update("./GUI_result_files/fig2.png")
            test.combine_data(CarbonFileName="NMR-1D.csv")
        except Exception as e:
             sg.popup(e)
    if event=="-submit3.5-":
        try:
            prameter=["AllDB","Plant","Human","Microorganism","Drug"]
            options=["OPTION"+str(i+1) for i in range(5) ]
            option_values=[values[i] for i in options]
            #读入Filter和evaluator
            select_para=[]
            for index,value in enumerate(option_values):
                if value==True:
                    select_para.append(prameter[index])
            dblink="Database"+"/"+select_para[0]+".npz"
            database1=np.load(dblink,allow_pickle=True)["data"][()]
            sg.popup("Successfully loaded database!")
            window["-submit3.5-"].update(button_color=('white', 'green'))
        except Exception as e:
             sg.popup(e)
    if event=="-submit3.6-":
        try:
            dblink=values['Other database']
            database1=np.load(dblink,allow_pickle=True)["data"][()]
            sg.popup("Successfully loaded database!")
            window["-submit3.6-"].update(button_color=('white', 'green'))
        except Exception as e:
             sg.popup(e)
        
    if event=="-submit4-":
        try:
            prameter=["AllDB","Plant","Human","Microorganism","Drug","CNF","CTNF","MW","CSS","AAS","FPS","FPAACS"]
            options=["OPTION"+str(i+1) for i in range(12) ]
            option_values=[values[i] for i in options]

            #读入Filter和evaluator
            select_para=[]
            for index,value in enumerate(option_values):
                if value==True:
                    select_para.append(prameter[index])

            Filters=select_para[1:-1]
            evaluator=select_para[-1]


            #读入待分析的数据
            df=pd.read_csv("NMR-1D.csv")
            shifts=df.iloc[:,0].tolist()
            Ctype=df.iloc[:,1].tolist()

            #读入GUI中的控制参数
            CarbonTypeNum=[Ctype.count("q"),Ctype.count("t"),Ctype.count("d"),Ctype.count("s")]
            CNFbias=int(values["-CNFbias-"])
            CTNFbias=int(values["-CTNFbias-"])
            FPAACSweights=values["-weights-"]
            FPAACSweights=np.array(FPAACSweights.split(','), dtype=float)
            FPAACSweights=list(FPAACSweights)
            
            MWlist=values["-MWlist-"]
            MWlist = np.array(MWlist.split(','), dtype=float)
            MWlist=list(MWlist)
            
            MWbias=5
            TopN=80
            
            task=Filters_and_evaluators(np.array(shifts),np.array(Ctype),database1)
            
            #加上CarbonNumFilter
            if "CNF" in Filters:
                print("CNF")
                task.CarbonNumFilter(bias=CNFbias)
            #加上CarbonTypeNumFilter
            if "CTNF" in Filters:
                print("CTNF")
                task.CarbonTypeNumFilter(CarbonTypeNum=CarbonTypeNum,bias=CTNFbias)
            #加上MWFilter
            if "MW" in Filters:
                print("MW")
                task.MWFilter(MWlist,bias=MWbias)
            
            if len(task.database["DBindex"])==0:
                sg.popup("No compounds meet the search criteria!")
            

            if evaluator=="FPS":
                print("FPS")
                task.FPS_evaluator()
                task.ShowTopN(task.FPSscore,TopN=TopN)
            if evaluator=="AAS" or evaluator=="CSS":
                task.CSS_AAS_evaluator()
                if evaluator=="AAS":
                    print("AAS")
                    task.ShowTopN(task.AASscore,TopN=TopN)
                if evaluator=="CSS":
                    print("CSS")
                    task.ShowTopN(task.CSSscore,TopN=TopN)
            if evaluator=="FPAACS":
                print("FPAACS")
                task.FPAACS_evaluator(weights=FPAACSweights)
                task.ShowTopN(task.FPAACSscore,TopN=TopN)
            
            result=task.TOPN
            result.to_csv("Result.csv")
            legends=["Score:"+str(round(task.TOPN["score"][i],2))+"/"+"ID:"+str(round(task.TOPN["DBindex"][i],2)) for i in range(len(task.TOPN["score"]))]
            plot_molecules_with_legend(task.TOPN["smiles"], legends,grid_width=4, image_size=(200, 200), output_filename="./GUI_result_files/fig4.png")

            window['-fig4-'].update("./GUI_result_files/fig4.png")
            sg.popup("Task Finish!")
        except Exception as e:
             sg.popup(e)
    if event=="-submit5-":
        
        try:
            
            index=int(values["-DBindex-"])
            DBindexlist=task.TOPN["DBindex"].tolist()
            rowindex=DBindexlist.index(index)
            shifts=list(task.TOPN["Vir_shifts"][rowindex])
            shifts=[round(i,1) for i in shifts]
            window['-shifts-'].update(shifts)
            window['-Ctype-'].update(list(task.TOPN["Ctype"][rowindex]))
            window['-smiles-'].update(task.TOPN["smiles"][rowindex]) 
        except Exception as e:
             sg.popup(e)
    if event=="-submit21-":
        try:
            sg.popup("The process of creating a database takes some time, please be patient and wait.")
            smileslink=values["-In21-"]
            df=pd.read_csv(smileslink)
            smiles1=df.iloc[:,0].tolist()
            VirDBGenerator(smiles1)
            sg.popup("Job finish.A file (generated-virDB.npz) has been created in the root directory.This file can be imported and used for structural identification ")
        except Exception as e:
             sg.popup(e)
#窗口切换
    if event=="-func1-":
        window["-element11-"].update(visible=True)
        window["-element12-"].update(visible=True)
        window["-element21-"].update(visible=False)
    if event=="-func2-":
        window["-element11-"].update(visible=False)
        window["-element12-"].update(visible=False)
        window["-element21-"].update(visible=True)
window.close()


# In[ ]:




