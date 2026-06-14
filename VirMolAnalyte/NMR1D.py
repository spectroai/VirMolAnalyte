#!/usr/bin/env python
# coding: utf-8

# In[1]:

import nmrglue as ng
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import heapq
import math
import os


def _default_gui_result_dir():
    """Default folder for NMRPipe .fid intermediates and GUI figures."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "GUI_result_files")


def _pipe_fid_path(name, pipe_output_dir):
    os.makedirs(pipe_output_dir, exist_ok=True)
    return os.path.join(pipe_output_dir, name)


class CNMR_peak(object):
    def __init__(self, dir_C, pipe_output_dir=None):
        self.dir_C = dir_C
        self.pipe_output_dir = pipe_output_dir or _default_gui_result_dir()
    def GetCpeak (self,y_up=2*1e8,threshold_C=0.05*1e8,x_left=200,x_right=-2,DrawWin="True"):             
        """Get the peak of the 13C NMR""" 
         # 原始数据读取
        dic, data = ng.bruker.read_pdata(self.dir_C, scale_data=False)
        self.threshold_C=threshold_C                  
        #设置文件参数
        udic = ng.bruker.guess_udic(dic, data)
        # 实现格式转换成NMRPipe文件，这个新文件的参数文件夹没有改变
        C = ng.convert.converter()
        C.from_bruker(dic, data, udic)
        fid_c = _pipe_fid_path("1d_pipe.fid", self.pipe_output_dir)
        ng.pipe.write(fid_c, *C.to_pipe(), overwrite=True)
        #重新读取新文件,获取单位转化对象
        self.dic, self.data = ng.pipe.read(fid_c)
        self.uc = ng.pipe.make_uc(self.dic, self.data)
        self.peaks = ng.peakpick.pick(self.data, pthres=threshold_C, algorithm="downward")
        # if DrawWin=="True":
        #     get_ipython().run_line_magic('matplotlib', 'qt')
        # else:
        #     get_ipython().run_line_magic('matplotlib', 'inline')
        #在原始图谱上查看峰提取效果(135)
        plt.figure(figsize=(4, 3))
        plt.plot(self.uc.ppm_scale(), data)
        # add markers for peak positions
        ppmcarbon=[]
        heightcarbon=[]
        for n, peak in enumerate(self.peaks):
            height = data[int(peak["X_AXIS"])]
            heightcarbon.append(height)
            ppm = self.uc.ppm(peak["X_AXIS"])
            ppmcarbon.append(ppm)
            plt.scatter(ppm, height, marker="o", color="r", s=100, alpha=0.5)
            #plt.text(ppm, height + 5e5, n + 1, ha="center", va="center")
        plt.hlines(threshold_C, *self.uc.ppm_limits(), linestyle="--", color="k")
        plt.xlim(x_left, x_right)
        plt.ylim(0, y_up)
        plt.show()
        self.ppmcarbon=ppmcarbon
        self.heightcarbon=heightcarbon
    def SolventRemove(self,C_list,height,solvent="Chloroform"):
        if solvent=="Chloroform":
            hit=[]
            hit_height=[]
            for i in range(len(C_list)):
                if 76.5<=C_list[i]<=77.5:
                    hit.append(C_list[i])
                    hit_height.append(height[i])
            C_list= [x for x in C_list if x not in hit] 
            height=[x for x in height if x not in hit_height]
        if solvent=="Methanol":
            hit=[]
            hit_height=[]
            for i in range(len(C_list)):
                if 48.2<=C_list[i]<=49.6:
                    hit.append(C_list[i])
                    hit_height.append(height[i])
            C_list= [x for x in C_list if x not in hit]
            height=[x for x in height if x not in hit_height]
        if solvent=="Pyridine":
            hit=[]
            hit_height=[]
            for i in range(len(C_list)):
                if 149.3<=C_list[i]<=150.3 or 135.0<=C_list[i]<=135.8 or 123.0<=C_list[i]<=123.9:
                    hit.append(C_list[i])
                    hit_height.append(height[i])
            C_list= [x for x in C_list if x not in hit]
            height=[x for x in height if x not in hit_height]
        return(C_list,height) 
        if solvent=="DMSO":
            print("DMSO")
            hit=[]
            hit_height=[]
            for i in range(len(C_list)):
                if 39<=C_list[i]<=40:
                    hit.append(C_list[i])
                    hit_height.append(height[i])
            C_list= [x for x in C_list if x not in hit]
            height=[x for x in height if x not in hit_height]
        return(C_list,height) 



        
#比较一个混合13C NMR化学位移的方法
def ShiftCompareOne(VecResult,peaks,therohold=0.8,method="shifts",mode="explt"):
    if mode=="explt":
        explt_list1=peaks.iloc[:,0].tolist()
        explt_list=explt_list1[:]
        explt_list_type1=peaks.iloc[:,1].tolist()
        explt_list_type= explt_list_type1[:] 
    if mode=="validation":
        explt_list=list(peaks[2])
        explt_list_type=list(peaks[3])
    data=VecResult
    hit_explt_all=[]
    hit_lib_all=[]
    delta=[]
    scores=[]
    MSEscores=[]
    if method=="shifts":
        #只比较化学位移
        for i in range(len(data["shifts"])):
            cmp_lib=data["shifts"][i]
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
            #计算单个化合物的匹配得分
            score=round(len(hit_lib)/LibCarbonNum*100,1)
            scores.append(score)
            #计算MSEscore值
            MSEscore=MinDeltaSum/LibCarbonNum
            MSEscore=(2*math.atan(1/MSEscore)/(math.pi))
            MSEscores.append(MSEscore*100)
            #第一次循环完成后，开始数据库中第二个化合物比较，重新读取数据
            if mode=="explt":
                explt_list=peaks[:].iloc[:,0].tolist()
            if mode=="validation":
                explt_list=list(peaks[:][2])
            hit_lib_all.append(hit_lib)
            hit_explt_all.append(hit_explt)
        #将化学匹配得分添加到结果文件中，以便于后续统计分析
        VecResult["ShiftScore"]=scores
        VecResult["MSEvalues"]=MSEscores
        #将化学位移匹配详细结果单独保存
        col_name=[]
        IDs=VecResult["IDs"]
        for i in range(len(hit_explt_all)):
            col_name.append(IDs[i])
            col_name.append("explt")
        hit_data=[]
        for i in range(len(hit_explt_all)):
            hit_data.append(hit_lib_all[i])
            hit_data.append(hit_explt_all[i])
        df=pd.DataFrame(hit_data)
        HitShiftData=pd.DataFrame(df.values.T,columns=col_name)

    elif method=="shifts_types":
        #比较化学位移和类型
        for i in range(len(data["shifts"])):
            cmp_lib=data["shifts"][i]
            cmp_lib_type=data["Ctype"][i]
            hit_lib=[]
            hit_explt=[]
            score=[]
            MinDeltaSum=0
            LibCarbonNum=len(cmp_lib)
            for j in range(LibCarbonNum):
                #计算数据库中一个化学位移值和explt_list的误差
                delta_all=[abs(cmp_lib[j]-m) for m in explt_list[:]]
                # #这里于只比较化学位移不同，需要先对explt_list进行碳类型的挑选
                # delta_all=[delta_all[i] for i in range(len(delta_all)) if cmp_lib_type[j]==explt_list_type[i]]
                min_delta=min(delta_all)
                try:
                    if min_delta<=therohold and explt_list_type[delta_all.index(min(delta_all))]==cmp_lib_type[j]:
                        hit=explt_list[delta_all.index(min(delta_all))]
                        hit_lib.append(cmp_lib[j])
                        hit_explt.append(hit)
                        explt_list_type.remove(explt_list_type[delta_all.index(min(delta_all))])
                        explt_list.remove(hit)
                    if min_delta>therohold and explt_list_type[delta_all.index(min(delta_all))]==cmp_lib_type[j]:
                        min_delta=min_delta
                except:
                    min_delta=5
                MinDeltaSum+=np.square(min_delta)
                #如果最小误差小于阈值，判断为1个hit


            #计算单个化合物的匹配得分
            score=round(len(hit_lib)/LibCarbonNum*100,1)
            scores.append(score)
            #计算MSEscore值
            MSEscore=MinDeltaSum/LibCarbonNum
            MSEscore=(2*math.atan(1/MSEscore)/(math.pi))
            MSEscores.append(MSEscore*100)
            #第一次循环完成后，开始数据库中第二个化合物比较，重新读取数据
            if mode=="explt":
                explt_list=peaks[:].iloc[:,0].tolist()
                explt_list_type=peaks[:].iloc[:,1].tolist()
            if mode=="validation":
                explt_list=list(peaks[:][2])
                explt_list_type=list(peaks[:][3])
            hit_lib_all.append(hit_lib)
            hit_explt_all.append(hit_explt)
        #将化学匹配得分添加到结果文件中，以便于后续统计分析
        VecResult["ShiftScore"]=scores
        VecResult["MSEvalues"]=MSEscores
        #将化学位移匹配详细结果单独保存
        col_name=[]
        IDs=VecResult["IDs"]
        for i in range(len(hit_explt_all)):
            col_name.append(IDs[i])
            col_name.append("explt")
        hit_data=[]
        for i in range(len(hit_explt_all)):
            hit_data.append(hit_lib_all[i])
            hit_data.append(hit_explt_all[i])
        df=pd.DataFrame(hit_data)
        HitShiftData=pd.DataFrame(df.values.T,columns=col_name)

    return(VecResult,HitShiftData)
    
class CNMR_process(object):
    def __init__(self, dir_C, dir_C135, dir_C90, pipe_output_dir=None):
        self.dir_C = dir_C
        self.dir_C135 = dir_C135
        self.dir_C90 = dir_C90
        self.pipe_output_dir = pipe_output_dir or _default_gui_result_dir()
    def get_13CDEPT_peak (self,threshold_C=0.05*1e8,
                          threshold_C135pos=0.2*1e8,threshold_C135neg=-0.2*1e8,
                          threshold_C90=0.2*1e8,x_left=200,x_right=-2,DrawWin="True"):             
        """Get the peak of the 13C DEPT NMR""" 
         # 原始数据读取
        dic, data = ng.bruker.read_pdata(self.dir_C, scale_data=False)
        dic135, data135 = ng.bruker.read_pdata(self.dir_C135, scale_data=False)
        dic90, data90 = ng.bruker.read_pdata(self.dir_C90, scale_data=False)
        self.threshold_C=threshold_C                  
        self.threshold_C135pos=threshold_C135pos
        self.threshold_C135neg=threshold_C135neg
        self.threshold_C90=threshold_C90    
        #设置文件参数
        udic = ng.bruker.guess_udic(dic, data)
        udic135 = ng.bruker.guess_udic(dic135, data135)
        udic90=ng.bruker.guess_udic(dic90, data90)
        # udic[0]['size']     = 2048
        # udic[0]['complex']  = True
        # udic[0]['encoding'] = 'direct'
        # udic[0]['sw']       = 10000.000
        # udic[0]['obs']      = 600.133
        # udic[0]['car']      = 4.773 * 600.133
        # udic[0]['label']    = '13C'

        # 实现格式转换成NMRPipe文件，这个新文件的参数文件夹没有改变
        C = ng.convert.converter()
        C.from_bruker(dic, data, udic)
        C135 = ng.convert.converter()
        C135.from_bruker(dic135, data135, udic135)
        C90 = ng.convert.converter()
        C90.from_bruker(dic90, data90, udic90)
        fid_c = _pipe_fid_path("1d_pipe.fid", self.pipe_output_dir)
        fid_c135 = _pipe_fid_path("1d_pipe135.fid", self.pipe_output_dir)
        fid_c90 = _pipe_fid_path("1d_pipe90.fid", self.pipe_output_dir)
        ng.pipe.write(fid_c, *C.to_pipe(), overwrite=True)
        ng.pipe.write(fid_c135, *C135.to_pipe(), overwrite=True)
        ng.pipe.write(fid_c90, *C90.to_pipe(), overwrite=True)

        #重新读取新文件,获取单位转化对象
        self.dic, self.data = ng.pipe.read(fid_c)
        self.uc = ng.pipe.make_uc(self.dic, self.data)
        self.dic135, self.data135 = ng.pipe.read(fid_c135)
        self.uc135 = ng.pipe.make_uc(self.dic135, self.data135)
        self.dic90, self.data90 = ng.pipe.read(fid_c90)
        self.uc90 = ng.pipe.make_uc(self.dic90, self.data90)
        
        self.peaks = ng.peakpick.pick(self.data, pthres=threshold_C, algorithm="downward")
        self.peaks135 = ng.peakpick.pick(self.data135, pthres=threshold_C135pos,nthres=threshold_C135neg, algorithm="downward")
        self.peaks90 = ng.peakpick.pick(self.data90, pthres=threshold_C90,algorithm="downward")
        # if DrawWin=="True":
        #     get_ipython().run_line_magic('matplotlib', 'qt')
        # elif DrawWin=="False":
        #     get_ipython().run_line_magic('matplotlib', 'inline')
        # else:
        #     pass
        #在原始图谱上查看峰提取效果(135)
        plt.figure(figsize=(9, 5))
        plt.subplot(3,1,1)
        plt.plot(self.uc135.ppm_scale(), data135)
        ppm135=[]
        height135=[]
        # add markers for peak positions
        for n, peak135 in enumerate(self.peaks135):
            height = data135[int(peak135["X_AXIS"])]
            height135.append(height)
            ppm = self.uc135.ppm(peak135["X_AXIS"])
            ppm135.append(ppm)
            plt.scatter(ppm, height, marker="o", color="r", s=100, alpha=0.5)
            #plt.text(ppm, height + 5e5, n + 1, ha="center", va="center")
        plt.hlines(threshold_C135pos, *self.uc.ppm_limits(), linestyle="--", color="k")
        plt.hlines(threshold_C135neg, *self.uc.ppm_limits(), linestyle="--", color="k")
        plt.xticks([])
        plt.xlim(x_left, x_right)
        
        plt.subplot(3,1,2)
        plt.plot(self.uc90.ppm_scale(), data90)
        # add markers for peak positions
        ppm90=[]
        height90=[]
        for n, peak90 in enumerate(self.peaks90):
            height = data90[int(peak90["X_AXIS"])]
            height90.append(height)
            ppm = self.uc90.ppm(peak90["X_AXIS"])
            ppm90.append(ppm)
            plt.scatter(ppm, height, marker="o", color="r", s=100, alpha=0.5)
            #plt.text(ppm, height + 5e5, n + 1, ha="center", va="center")
        plt.hlines(threshold_C90, *self.uc.ppm_limits(), linestyle="--", color="k")
        plt.xlim(x_left, x_right)
        plt.xticks([])

        plt.subplot(3,1,3)
        plt.plot(self.uc.ppm_scale(), data)
        # add markers for peak positions
        ppmcarbon=[]
        heightcarbon=[]
        for n, peak in enumerate(self.peaks):
            height = data[int(peak["X_AXIS"])]
            heightcarbon.append(height)
            ppm = self.uc.ppm(peak["X_AXIS"])
            ppmcarbon.append(ppm)
            plt.scatter(ppm, height, marker="o", color="r", s=100, alpha=0.5)
            #plt.text(ppm, height + 5e5, n + 1, ha="center", va="center")
        plt.hlines(threshold_C, *self.uc.ppm_limits(), linestyle="--", color="k")
        plt.xlim(x_left, x_right)
        plt.ylim(0, 1e8)
        plt.savefig(r"./GUI_result_files/fig1.png")
        # plt.show()
        self.ppm135=ppm135
        self.ppm90=ppm90
        self.ppmcarbon=ppmcarbon
        self.height135=height135
        self.height90=height90
        self.heightcarbon=heightcarbon
        

    def sort_Ctype_mindelta(self,therohold=0.2): 
        """Provides methods to classify and count carbons"""
        list_NMR=self.ppmcarbon
        list_90=self.ppm90
        list_135_CH2=[self.ppm135[i] for i in range(len(self.ppm135))if self.height135[i]<0]
        list_135_CH_CH3=[self.ppm135[i] for i in range(len(self.ppm135))if self.height135[i]>0]
        self.hit_CH3=[]
        self.hit_CH2=[]
        self.hit_CH=[]
        
        for a in list_135_CH2:
            lst=[]
            for b in list_NMR:
                lst.append(abs(a-b))
            hit=list_NMR[lst.index(min(lst))]
            if min(lst)<=therohold:
                self.hit_CH2.append(hit)
      
        list_NMR= [x for x in list_NMR if x not in self.hit_CH2]
        
        

        for a in list_90:
            lst=[]
            for b in list_NMR:
                lst.append(abs(a-b))
            hit=list_NMR[lst.index(min(lst))]
            if min(lst)<=therohold:
                self.hit_CH.append(hit)

        list_NMR= [x for x in list_NMR if x not in self.hit_CH]



        hit_total=[]
        for a in list_90:
            lst=[]
            for b in list_135_CH_CH3:
                lst.append(abs(a-b))
            hit=list_135_CH_CH3[lst.index(min(lst))]
            if min(lst)<=therohold:
                hit_total.append(hit)
        list_135_CH_CH3= [x for x in list_135_CH_CH3 if x not in hit_total]
        
        for a in list_135_CH_CH3:
            lst=[]
            for b in list_NMR:
                lst.append(abs(a-b))
            hit=list_NMR[lst.index(min(lst))]
            if min(lst)<=therohold:
                self.hit_CH3.append(hit)

        self.Hit_C_or_unhited= [x for x in list_NMR if x not in self.hit_CH3]
#-------------------------绘图------------------------------------------------------
    def CNMR_reconstract(self,width=2,fontsize=30,chemical_shift="False",
                         label="False",summary="False",DrawWin="True",x_left=200,x_right=-2,
                         y_top=0.6*1e8,y_bottom=-0.6*1e8):
        # if DrawWin=="True":
        #     get_ipython().run_line_magic('matplotlib', 'qt')
        # else:
        #     get_ipython().run_line_magic('matplotlib', 'inline')
        num=0
        plt.figure(figsize=(9, 3))
        #plt.xlabel('ppm', fontsize =10)
        #plt.ylabel('Hight', fontsize =10)
        #plt.tick_params(labelsize=10)
        for n, ppm in enumerate(self.hit_CH):
            num=num+1
            index=self.ppmcarbon.index(ppm)
            height = self.heightcarbon[index]
            plt.bar(ppm, height,width=width,color="b")
            if chemical_shift=="True":
                plt.text(ppm,height,round(ppm,1), ha="center", va="center",rotation=90,
                         color="black",size=fontsize)
            if label=="True":
                plt.text(ppm,height,n, ha="center", va="center",rotation=90,
                color="black",size=fontsize)

        for n, ppm in enumerate(self.hit_CH3):
            num=num+1
            index=self.ppmcarbon.index(ppm)
            height = self.heightcarbon[index]
            plt.bar(ppm, height,width=width,color="r")
            if chemical_shift=="True":
                plt.text(ppm,height,round(ppm,1), ha="center", va="center",rotation=90,
                     size=fontsize,color="black")
            if label=="True":
                plt.text(ppm,height,n, ha="center", va="center",rotation=90,
                         color="black",size=fontsize)
        for n, ppm in enumerate(self.hit_CH2):
            num=num+1
            index=self.ppmcarbon.index(ppm)
            height = self.heightcarbon[index]
            plt.bar(ppm, -height,width=width,color="g")
            if chemical_shift=="True":
                plt.text(ppm,-height,round(ppm,1), ha="center", va="center",rotation=90,
                 fontsize=fontsize,color="black")
            if label=="True":
                plt.text(ppm,-height,n, ha="center", va="center",rotation=90,
                         color="black",fontsize=fontsize)
        for n, ppm in enumerate(self.Hit_C_or_unhited):
            num=num+1
            index=self.ppmcarbon.index(ppm)
            height = self.heightcarbon[index]
            plt.bar(ppm, -height,width=width,color="orange")
            if chemical_shift=="True":
                plt.text(ppm,-height,round(ppm,1), ha="center", va="center",rotation=90,
                fontsize=fontsize,color="black")
            if label=="True":
                plt.text(ppm,-height,n, ha="center", va="center",rotation=90,
                         color="black",fontsize=fontsize)
        plt.hlines(0, *self.uc.ppm_limits(), linestyle="-", color="black")
        plt.xlim(x_left, x_right)

        x0, xmax = plt.xlim()
        y0, ymax = plt.ylim()
        data_width = abs(xmax - x0)
        data_height = abs(ymax)
        plt.savefig(r"./GUI_result_files/fig2.png")
        # plt.show()
        
        if summary=="True":
            sum=[]
            col=["CH3:"+str(len(self.hit_CH3)),"CH2:"+str(len(self.hit_CH2)),
                 "CH:"+str(len(self.hit_CH)),"C:"+str(len(self.Hit_C_or_unhited)),"Total"]
            df=pd.DataFrame(sum)
            CH3_list=[round(peak,1)for peak in self.hit_CH3] 
            CH2_list=[round(peak,1)for peak in self.hit_CH2]
            CH_list=[round(peak,1)for peak in self.hit_CH]
            C_or_unhited_list=[round(peak,1)for peak in self.Hit_C_or_unhited]
            sum.append(CH3_list)
            sum.append(CH2_list)
            sum.append(CH_list)
            sum.append(C_or_unhited_list)
            sum.append([round(len(self.hit_CH3)+len(self.hit_CH2)+len(self.hit_CH)+len(self.Hit_C_or_unhited),0)])
            df=pd.DataFrame(sum)
            self.summary=pd.DataFrame(df.values.T,columns=col)
#------------------------去除杂质和溶剂---------------------------------------------
    def impurity_removal(self,C_list,type="C",rate=0.5,Standsrd_num=0,method="relative",threshold=0.2*1e7):
        del_index=[]
        for n, peak in enumerate(C_list):
            index=self.ppmcarbon.index(peak)
            height = self.heightcarbon[index]
            if method=="relative":
                if abs(height-standard_height)/standard_height>=threshold:
                    del_index.append(n)
            if method=="absolute":
                if abs(height)-threshold<=0:
                    del_index.append(n)
        del_index.reverse()
        for i in del_index:
            del C_list[i]
        if type=="CH3":
            self.hit_CH3=C_list
        if type=="CH2":
            self.hit_CH2=C_list
        if type=="CH":
            self.hit_CH=C_list
        if type=="C":
            self.Hit_C_or_unhited=C_list
        return(C_list)
    def solvent_remove(self,C_list,type="C",solvent="Chloroform"):
        if solvent=="Chloroform":
            hit=[]
            for i in range(len(C_list)):
                if 76.5<=C_list[i]<=77.5:
                    hit.append(C_list[i])
            C_list= [x for x in C_list if x not in hit] 
        if solvent=="Methanol":
            hit=[]
            for i in range(len(C_list)):
                if 48.2<=C_list[i]<=49.6:
                    hit.append(C_list[i])
            C_list= [x for x in C_list if x not in hit]
        if solvent=="Pyridine":
            hit=[]
            for i in range(len(C_list)):
                if 149.3<=C_list[i]<=150.3 or 135.0<=C_list[i]<=135.8 or 123.0<=C_list[i]<=123.9:
                    hit.append(C_list[i])
            C_list= [x for x in C_list if x not in hit]
        if solvent=="DMSO":
            hit=[]
            for i in range(len(C_list)):
                if 38.8<=C_list[i]<=40.2:
                    hit.append(C_list[i])
            C_list= [x for x in C_list if x not in hit]
        if type=="CH3":
            self.hit_CH3=C_list
        if type=="CH2":
            self.hit_CH2=C_list
        if type=="CH":
            self.hit_CH=C_list
        if type=="C":
            self.Hit_C_or_unhited=C_list
#-------------------------碳谱信息储存--------------------------------------------
    def combine_data(self,CarbonFileName):
        C_shift=[]
        C_type=[]
        C_height=[]
        for i in self.hit_CH3:
            C_shift.append(round(i,2))
            height=self.heightcarbon[self.ppmcarbon.index(i)]
            C_type.append("q")
            C_height.append(height)
        for i in self.hit_CH2:
            C_shift.append(round(i,2))
            height=self.heightcarbon[self.ppmcarbon.index(i)]
            C_type.append("t")
            C_height.append(height)
        for i in self.hit_CH:
            C_shift.append(round(i,2))
            height=self.heightcarbon[self.ppmcarbon.index(i)]
            C_type.append("d")
            C_height.append(height)
        for i in self.Hit_C_or_unhited:
            C_shift.append(round(i,2))
            height=self.heightcarbon[self.ppmcarbon.index(i)]
            C_type.append("s")
            C_height.append(height)
        self.C_shift=C_shift
        self.C_type=C_type
        self.C_height=C_height
        df=pd.DataFrame([C_shift,C_type,C_height])
        combine=pd.DataFrame(df.values.T)
        self.Explt_data=combine
        combine.to_csv(CarbonFileName,index=None)
        self.explt_dir=CarbonFileName
#----------------------碳谱信息和数据库比较-------------------------------------------------
    #在新的比较方法中，选择满足阈值的最近的，作为匹配值
    def carbon_compare_new(self,FileName, C_DEPT_data,CompareResult,therohold=0.8,DrawWin="False",fontsize=10):
        """
        CompareResult:the save pathway of the analysis result
        """
        # if DrawWin=="True":
        #     get_ipython().run_line_magic('matplotlib', 'qt')
        # else:
        #     get_ipython().run_line_magic('matplotlib', 'inline')
        df=pd.read_csv(FileName)
        explt_dir=FileName
        explt_list=df.iloc[:,0].tolist()
        hit_explt_all=[]
        hit_lib_all=[]
        delta=[]
        scores=[]
        #只比较化学位移
        for i in range(int(C_DEPT_data.shape[1]/2)):
            cmp_lib1=C_DEPT_data.iloc[:,2*i].tolist()
            cmp_lib=[]
            
            for elem in cmp_lib1:
                if not np.isnan(elem):
                    cmp_lib.append(elem)
            hit_lib=[]
            hit_explt=[]
            score=[]
            for j in range(len(cmp_lib)):
                delta_all=[]
                for m in explt_list[:]:
                    delta=abs(cmp_lib[j]-m)
                    delta_all.append(delta)
                if min(delta_all)<=therohold:
                    hit=explt_list[delta_all.index(min(delta_all))]
                    hit_lib.append(cmp_lib[j])
                    hit_explt.append(hit)
                    explt_list.remove(hit)
            #第一次循环完成后，开始数据库中第二个化合物比较，重新读取数据
            b=pd.read_csv(explt_dir)
            explt_list=b.iloc[:,0].tolist()
            score=round(len(hit_lib)/len(cmp_lib)*100,1)
            scores.append(score)
            hit_lib_all.append(hit_lib)
            hit_explt_all.append(hit_explt)

        #生成列名
        col_name=[]
        for i in range(len(hit_explt_all)):
            col_name.append("comp"+str(i+1))
            col_name.append("explt")

        #合并匹配上的数据
        hit_data=[]
        for i in range(len(hit_explt_all)):
            hit_data.append(hit_lib_all[i])
            hit_data.append(hit_explt_all[i])
        df=pd.DataFrame(hit_data)
        self.hit_data=pd.DataFrame(df.values.T,columns=col_name)
        self.hit_data.to_csv(CompareResult,index=None)
        self.scores=scores
        #获取分数前10的化合物，并且可视化
        #首先获取最大的十个化合物的索引和分数

        testList =scores
        tmp = zip(range(len(testList)), testList)
        large10 = heapq.nlargest(10, tmp, key=lambda x:x[1])
        #实现可视化
        plt.figure(figsize=(4,3))
        for i in range(len(large10)):
            x="Cmp"+str(large10[i][0]+1)
            y=large10[i][1]
            plt.bar(x,y,color="black")
            plt.text(x,y+0.5,y,ha='center',color="red")
            plt.xticks(rotation=45)
        plt.ylabel('Score', fontsize =fontsize)
        plt.tick_params(labelsize=fontsize)
        plt.savefig(r"./FilesFromGUI/fig3.png",bbox_inches = 'tight')

    def carbon_compare_type_new(self,CarbonFileName, C_DEPT_data,CompareResult,therohold=0.8,DrawWin="False",fontsize =10):
        # if DrawWin=="True":
        #     get_ipython().run_line_magic('matplotlib', 'qt')
        # else:
        #     get_ipython().run_line_magic('matplotlib', 'inline')
        df=pd.read_csv(CarbonFileName)
        explt_dir=CarbonFileName
        explt_list=df.iloc[:,0].tolist()
        explt_list_type=df.iloc[:,1].tolist()
        hit_explt_all=[]
        hit_explt_type_all=[]
        hit_lib_all=[]
        hit_lib_type_all=[]
        delta=[]
        scores=[]
        #只比较化学位移
        for i in range(int(C_DEPT_data.shape[1]/2)):
            cmp_lib1=C_DEPT_data.iloc[:,2*i].tolist()
            cmp_lib_type1=C_DEPT_data.iloc[:,2*i+1].tolist()
            cmp_lib=[]
            for elem in cmp_lib1:
                if not np.isnan(elem):
                    cmp_lib.append(elem)
            cmp_lib_type=cmp_lib_type1[0:len(cmp_lib)]
            hit_lib=[]
            hit_lib_type=[]
            hit_explt=[]
            hit_explt_type=[]
            score=[]
            for j in range(len(cmp_lib)):
                delta_all=[]
                for m,peak in enumerate(explt_list[:]):
                    delta=abs(cmp_lib[j]-peak)
                    delta_all.append(delta)
                    index_min=delta_all.index(min(delta_all))
                if min(delta_all)<=therohold and cmp_lib_type[j]==explt_list_type[index_min]:
                    hit_lib.append(cmp_lib[j])
                    hit_lib_type.append(cmp_lib_type[j])
                    hit=explt_list[index_min]
                    hit_explt.append(hit)
                    hit_explt_type.append(explt_list_type[index_min])
                    explt_list.remove(hit)
                    explt_list_type.remove(explt_list_type[index_min])
            b=pd.read_csv(explt_dir)
            explt_list=b.iloc[:,0].tolist()
            explt_list_type=b.iloc[:,1].tolist()
            score=round(len(hit_lib)/len(cmp_lib)*100,1)
            scores.append(score)
            hit_lib_all.append(hit_lib)
            hit_lib_type_all.append(hit_lib_type)
            hit_explt_all.append(hit_explt)
            hit_explt_type_all.append(hit_explt_type)

        #生成列名
        col_name=[]
        for i in range(len(hit_explt_all)):
            col_name.append("comp"+str(i+1))
            col_name.append("explt")
            col_name.append("type")
                #生成列名
        col_name1=[]
        for i in range(len(hit_explt_all)):
            col_name1.append("comp"+str(i+1))
            col_name1.append("explt")

        #合并匹配上的数据
        hit_data=[]
        hit_data1=[]#不包括碳类型的单独存一个表格，为了方便后面混合物分析
        for i in range(len(hit_explt_all)):
            hit_data.append(hit_lib_all[i])
            hit_data.append(hit_explt_all[i])
            hit_data.append(hit_explt_type_all[i])
            hit_data1.append(hit_lib_all[i])
            hit_data1.append(hit_explt_all[i])
        df=pd.DataFrame(hit_data)
        df1=pd.DataFrame(hit_data1)
        hit_data=pd.DataFrame(df.values.T,columns=col_name)
        hit_data1=pd.DataFrame(df1.values.T,columns=col_name1)
        self.hit_data=hit_data
        self.hit_data1=hit_data1
        self.hit_data1.to_csv(CompareResult,index=None)
        self.scores_type=scores
        #获取分数前10的化合物，并且可视化
        #首先获取最大的十个化合物的索引和分数
        
        testList =scores
        tmp = zip(range(len(testList)), testList)
        large10 = heapq.nlargest(10, tmp, key=lambda x:x[1])
        #实现可视化
        plt.figure(figsize=(4,3))
        for i in range(len(large10)):
            x="Cmp"+str(large10[i][0]+1)
            y=large10[i][1]
            plt.bar(x,y,color="black")
            plt.text(x,y+0.5,y,ha='center',fontsize=fontsize,color="red")
            plt.xticks(rotation=45,fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
        plt.savefig(r"./FilesFromGUI/fig3.png",bbox_inches = 'tight')

#----------------------与.npz格式的数据进行比较的方法,这个方法相当慢---------------------------------------------------------
        
    def carbon_compare_db(self,FileName, CompareResult,data=r'./database/shiftdata.npz',min_carbon=10,therohold=0.8,DrawWin="False"):
        """
        FileName:the csv file of the experiemnt nmr data
        compare the data from nmrshitftdb2
        CompareResult:the save pathway of the analysis result
        data: the passway of .npz file
        """
        data=np.load(data, allow_pickle=True)['data'][()]
        # if DrawWin=="True":
        #     get_ipython().run_line_magic('matplotlib', 'qt')
        # else:
        #     get_ipython().run_line_magic('matplotlib', 'inline')
        df=pd.read_csv(FileName)
        explt_dir=FileName
        explt_list=df.iloc[:,0].tolist()
        hit_explt_all=[]
        hit_lib_all=[]
        delta=[]
        scores=[]
        #只比较化学位移
        for i in range(len(data["shifts"])):
            cmp_lib=data["shifts"][i]
            if len(cmp_lib)>= min_carbon:
                hit_lib=[]
                hit_explt=[]
                score=[]
                for j in range(len(cmp_lib)):
                    delta_all=[]
                    for m in explt_list[:]:
                        delta=abs(cmp_lib[j]-m)
                        delta_all.append(delta)
                    if min(delta_all)<=therohold:
                        hit=explt_list[delta_all.index(min(delta_all))]
                        hit_lib.append(cmp_lib[j])
                        hit_explt.append(hit)
                        explt_list.remove(hit)
                #第一次循环完成后，开始数据库中第二个化合物比较，重新读取数据
                b=pd.read_csv(explt_dir)
                explt_list=b.iloc[:,0].tolist()
                score=round(len(hit_lib)/len(cmp_lib)*100,1)
                scores.append(score)
                hit_lib_all.append(hit_lib)
                hit_explt_all.append(hit_explt)
            else:
                b=pd.read_csv(explt_dir)
                explt_list=b.iloc[:,0].tolist()
                hit_lib_all.append(["NUM"])
                hit_explt_all.append("NUM")
                scores.append(0)
            if (i+1) % 1000 == 0: print('%d/%d processed' %(i+1, len(data["shifts"])))

        #生成列名
        col_name=[]
        for i in range(len(hit_explt_all)):
            col_name.append("comp"+str(i))
            col_name.append("explt")

        #合并匹配上的数据
        hit_data=[]
        for i in range(len(hit_explt_all)):
            hit_data.append(hit_lib_all[i])
            hit_data.append(hit_explt_all[i])
        df=pd.DataFrame(hit_data)
        self.hit_data=pd.DataFrame(df.values.T,columns=col_name)
        self.hit_data.to_csv(CompareResult,index=None)
        self.scoresDB2=scores
    #     self.hit_data=pd.DataFrame(df.values.T,columns=col_name)
    #     self.hit_data.to_csv(CompareResult,index=None)
        #获取分数前10的化合物，并且可视化
        #首先获取最大的十个化合物的索引和分数

        testList =scores
        tmp = zip(range(len(testList)), testList)
        large10 = heapq.nlargest(10, tmp, key=lambda x:x[1])
        #实现可视化
        plt.figure(figsize=(5.4,3.5))
        for i in range(len(large10)):
            x="Cmp"+str(large10[i][0])
            y=large10[i][1]
            plt.bar(x,y,color="black")
            plt.text(x,y+0.5,y,ha='center',color="red")
            plt.xticks(rotation=45)
        plt.ylabel('Score', fontsize =15)
        plt.tick_params(labelsize=15)
        plt.savefig("fig3.png",bbox_inches = 'tight')
#选择任意两个化合物进行比较，并且作图
    def compare_select(self,num,database,fontsize =10):
        #获取目标化合物的全部数据
        explt_data=pd.read_csv(self.explt_dir)
        explt_shift_data=explt_data.iloc[:,0]
        explt_type_data=explt_data.iloc[:,1]
        #获取数据库内对应化合物的数据
        lib_shift_data=database.iloc[:,num*2-2:num*2-1].iloc[:,0].tolist()
        lib_type_data=database.iloc[:,num*2-1:num*2].iloc[:,0].tolist()
        #获取匹配上的目标化合物的碳信号：
        explt_shift=self.hit_data.iloc[:,int(num*2-1):int(num*2)].iloc[:,0].tolist()
        #获取匹配上的数据库内对应化合物的碳信号：
        lib_shift=self.hit_data.iloc[:,int(num*2-2):int(num*2-1)].iloc[:,0].tolist()

        #根据化合物全部信息绘图，这里高度信息已经没有了
        plt.figure(figsize=(9, 6))
        height=10
        width=0.6
        plt.subplot(2,1,1)
        for n,peak in enumerate(explt_shift_data):
            a=peak in explt_shift
            if a==False and str(explt_type_data[n])=="q":
                plt.scatter(peak, height, marker="o", color="r", s=100, alpha=0.5)
                plt.text(peak,height+3,peak,ha="center", va="center",rotation=90)
            if a==False and str(explt_type_data[n])=="d":
                plt.scatter(peak, height, marker="o", color="r", s=100, alpha=0.5)
                plt.text(peak,height+3,peak,ha="center", va="center",rotation=90)
            if a==False and str(explt_type_data[n])=="t":
                plt.scatter(peak, -height, marker="o", color="r", s=100, alpha=0.5)
                plt.text(peak,-height-3,peak,ha="center", va="center",rotation=90)
            if a==False and str(explt_type_data[n])=="s":
                plt.scatter(peak, -height, marker="o", color="r", s=100, alpha=0.5)
                plt.text(peak,-height-3,peak,ha="center", va="center",rotation=90)
            if str(explt_type_data[n])=="q": 
                    plt.bar(peak, height=height,width=width,color="r")

            if str(explt_type_data[n])=="t": 
                plt.bar(peak, height=-height,width=width,color="green")

            if str(explt_type_data[n])=="d": 
                plt.bar(peak, height=height,width=width,color="blue")

            if str(explt_type_data[n])=="s": 
                plt.bar(peak, height=-height,width=width,color="orange")
        plt.xlim(200,0)
        plt.ylim(-15,15)
        plt.xticks(fontsize =fontsize)
        plt.yticks(fontsize =fontsize)
        plt.hlines(0,xmin=0,xmax=200,linestyle="-", color="black")

        #根据根据数据库中指定化合物全部信息绘图，这里高度信息已经没有了
        plt.subplot(2,1,2)
        for n,peak in enumerate(lib_shift_data):
            a=peak in lib_shift
        #     if a==False and str(lib_type_data[n])=="q":
        #         plt.scatter(peak, height, marker="o", color="r", s=100, alpha=0.5)
        #         plt.text(peak,height+3,peak,ha="center", va="center",rotation=90)
        #     if a==False and str(lib_type_data[n])=="d":
        #         plt.scatter(peak, height, marker="o", color="r", s=100, alpha=0.5)
        #         plt.text(peak,height+3,peak,ha="center", va="center",rotation=90)
        #     if a==False and str(lib_type_data[n])=="t":
        #         plt.scatter(peak, -height, marker="o", color="r", s=100, alpha=0.5)
        #         plt.text(peak,-height-3,peak,ha="center", va="center",rotation=90)
        #     if a==False and str(lib_type_data[n])=="s":
        #         plt.scatter(peak, -height, marker="o", color="r", s=100, alpha=0.5)
        #         plt.text(peak,-height-3,peak,ha="center", va="center",rotation=90)
            if str(lib_type_data[n])=="q": 
                plt.bar(peak, height=height,width=width,color="r")

            if str(lib_type_data[n])=="t": 
                plt.bar(peak, height=-height,width=width,color="green")

            if str(lib_type_data[n])=="d": 
                plt.bar(peak, height=height,width=width,color="blue")

            if str(lib_type_data[n])=="s": 
                plt.bar(peak, height=-height,width=width,color="orange")      
        plt.xlim(200,0)
        plt.ylim(-15,15)
        plt.xticks(fontsize =fontsize)
        plt.yticks(fontsize =fontsize)
        plt.hlines(0,xmin=0,xmax=200,linestyle="-", color="black")
        plt.xlabel('ppm', fontsize =fontsize)
        plt.savefig(r"./FilesFromGUI/fig4.png")
        plt.show()
        
    
#---------------------------混合物分析(分两次分析)------------------------------------------------
    def MoveMainComp(self,Maincompfile):
        """从混合物中减去主要化合物，实现准确的图谱拆分"""
        #在调用此函数前首先通过常规的方法获得所有化合物的信号
        #然后调用储存有主要化合物13C谱pts信号的csv文件
        #将13C谱信号相减得到去掉主要化合物后的self.peaks
        df=pd.read_csv(Maincompfile)
        ShiftPts=df.iloc[:,2].tolist()
#         for i in ShiftPts:
#             for j in self.peaks[]
        peaks=[i for i in self.peaks if i[0] not in ShiftPts]
        self.peaks=np.array(peaks)
        #再将碳信号进行分类
        #然后进行后续分析
    def MoveKnownComp(self,num,CompareResult,CarbonFileName):
        """
        num:要去掉的化合物
        CompareResult：上一次比较结果的CSV文件
        CarbonFileName：上一次提取的碳化学位移信息
        """
        #首先获得要去掉的化合物匹配上的信号
        df=pd.read_csv(CompareResult)
        Hitppm=df.iloc[:,2*num-1]
        Hitppm=[i for i in Hitppm if i==i]
        #然后获得上一次提取的碳化学位移信息
        #并根据化学位移找到对应的pts值，重新组合成列表
        #该列表即为这次分析中要删除掉的信号
        df1=pd.read_csv(CarbonFileName)
        ppm=df1.iloc[:,0].tolist()
        pts=df1.iloc[:,2].tolist()
        deletepts=[]
        for i in Hitppm:
            for j,peak in enumerate(ppm):
                if i==peak:
                    deletepts.append(pts[j])
        #现在从self.peaks中去掉dletepts
        peaks=[i for i in self.peaks if i[0] not in deletepts]
        self.peaks=np.array(peaks)
    def split(self,ShiftDataFile,threshold_C=4*1e6,threshold_CHCH2CH3=4*1e6,width=0.5,fontsize=10,x_left=200,x_right=0,shift="False",adjust="False"):
        #获得待分析化合物的化学位移数据、碳类型数据和pts数据
        explt_data=pd.read_csv(ShiftDataFile)
        explt_shift_data=explt_data.iloc[:,0].tolist()
        explt_type_data=explt_data.iloc[:,1].tolist()
        explt_height_data=explt_data.iloc[:,2].tolist()

        explt_shift_remain2=[]
        explt_type_remain2=[]
        explt_height_remain2=[]
        explt_shift_split=[]
        explt_type_split=[]
        explt_height_split=[]
        for i,height in enumerate(explt_height_data):
            if explt_type_data[i]=="s"and height>=threshold_C:
                explt_shift_split.append(explt_shift_data[i])
                explt_type_split.append(explt_type_data[i])
                explt_height_split.append(explt_height_data[i])
            elif height>=threshold_CHCH2CH3:
                explt_shift_split.append(explt_shift_data[i])
                explt_type_split.append(explt_type_data[i])
                explt_height_split.append(explt_height_data[i])
            else:
                explt_shift_remain2.append(explt_shift_data[i])
                explt_type_remain2.append(explt_type_data[i])
                explt_height_remain2.append(explt_height_data[i])
        #绘图
        plt.figure(figsize=(10, 3), dpi=300)
        text_height=[]
        for i,Ctype in enumerate(explt_type_remain2):
            height =explt_height_remain2[i]
            ppm=explt_shift_remain2[i]

            if Ctype=="q":
                plt.bar(ppm, height,width=width,color="red")
                text_height.append(height)
            if Ctype=="t":
                plt.bar(ppm, -height,width=width,color="g")
                text_height.append(-height)
            if Ctype=="d":
                plt.bar(ppm, height,width=width,color="b")
                text_height.append(height)
            if Ctype=="s":
                plt.bar(ppm, -height,width=width,color="orange")
                text_height.append(-height)
        if shift=="True":
            texts=[plt.text(explt_shift_remain2[i],text_height[i],round(explt_shift_remain2[i],1),ha="center", va="center",rotation=90) for i in range(len(explt_shift_remain2))]
            if adjust=="True":
                adjust_text(texts, 
                    arrowprops=dict(arrowstyle='->',lw= 1,color='gray'),
                              only_move={'text': 'x'},
                             expand_text=(3, 1),
                            #expand_objects=(3, 1.2),
                            force_text=(0.75, 0), 
                            force_objects=(1, 0),
                    )
        plt.hlines(0,0,200,linestyle="-", color="black")
        plt.xlim(x_left, x_right)
        #plt.ylim(y_bottom,y_top)
        plt.xlabel('ppm', fontsize =15)
        #plt.ylabel('Hight', fontsize =15)
        plt.tick_params(labelsize=15)
        plt.show()

        plt.figure(figsize=(10, 3), dpi=300)
        text_height1=[]
        for i,Ctype in enumerate(explt_type_split):
            height =explt_height_split[i]
            ppm=explt_shift_split[i]
            if Ctype=="q":
                plt.bar(ppm, height,width=width,color="red")
                text_height1.append(height)
            if Ctype=="t":
                plt.bar(ppm, -height,width=width,color="g")
                text_height1.append(-height)
            if Ctype=="d":
                plt.bar(ppm, height,width=width,color="b")
                text_height1.append(height)
            if Ctype=="s":
                plt.bar(ppm, -height,width=width,color="orange")
                text_height1.append(-height)
        if shift=="True":
            texts=[plt.text(explt_shift_split[i],text_height1[i],round(explt_shift_split[i],1),ha="center", va="center",rotation=90) for i in range(len(explt_shift_remain2))]
            if adjust=="True":
                adjust_text(texts, 
                    arrowprops=dict(arrowstyle='->',lw= 1,color='gray'),
                              only_move={'text': 'x'},
                             expand_text=(3, 1),
                            #expand_objects=(3, 1.2),
                            force_text=(0.75, 0), 
                            force_objects=(1, 0),)
        plt.hlines(0,0,200,linestyle="-", color="black")
        plt.xlim(x_left, x_right)
        #plt.ylim(y_bottom,y_top)
        plt.xlabel('ppm', fontsize =15)
        #plt.ylabel('Hight', fontsize =15)
        plt.tick_params(labelsize=15)
        plt.show()
        split1=[]
        split2=[]
        split1.append(explt_shift_split)
        split1.append(explt_type_split)
        split1.append(explt_height_split)
        split2.append(explt_shift_remain2)
        split2.append(explt_type_remain2)
        split2.append(explt_height_remain2)
        df1=pd.DataFrame(split1)
        df1=pd.DataFrame(df1.values.T)
        df2=pd.DataFrame(split2)
        df2=pd.DataFrame(df2.values.T)
        return(df1,df2)
    
