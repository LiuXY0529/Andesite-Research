# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:22:37 2024

@author: liuxinyu
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

#创建一个文件上传器让用户上传csv文件
uploaded_file = st.file_uploader("Please upload your data", type=['csv'])
if uploaded_file is not None:
    #读取CSV文件
    Data_test_all = pd.read_csv(uploaded_file)

    #删除缺失率>50%的样品数据
    Data_test_value = Data_test_all[['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'CaO', 'MgO', 'MnO', 'K2O', 'Na2O', 'P2O5',
             'Sc', 'V', 'Cr', 'Co','Ni','Cu','Zn', 
             'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu','Gd', 'Tb','Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y',
             'Nb','Ta','Zr','Hf',
             'Rb','Cs', 'Sr','Ba', 'Pb', 'Th', 'U']]
    #转置数据，一个元素一行
    Data_test_all = Data_test_all.T
    Data_test_value = Data_test_value.T
    #计算数据缺失率
    missing_rate = Data_test_value.isnull().sum()/Data_test_value.iloc[:,0].size
    dict_missing = {'element':missing_rate.index,'missing rate':missing_rate.values}
    missing = pd.DataFrame(dict_missing)
    missing_drop = missing[missing['missing rate']<0.5]
    Data_test_all = Data_test_all[missing_drop.element]
    Data_test_all = Data_test_all.T
    
    #重新计算主量元素数据
    elements = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'CaO', 'MgO', 'MnO', 'K2O', 'Na2O', 'P2O5']
    Major = Data_test_all[elements] #原主量元素数据
    Sum_Major = np.sum(Major,axis=1)
    Sum_Major = pd.DataFrame(Sum_Major)
    Major_new = 100*Major/Sum_Major.values #新主量元素数据
    for i in range(10):
        Data_test_all[elements[i]]=Major_new[elements[i]]#用新数据替换原数据
        
    #随机森林分类器预测安山岩构造环境
    Data_test_value = Data_test_all[['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'CaO', 'MgO', 'MnO', 'K2O', 'Na2O', 'P2O5',
             'Sc', 'V', 'Cr', 'Co','Ni','Cu','Zn', 
             'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu','Gd', 'Tb','Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y',
             'Nb','Ta','Zr','Hf',
             'Rb','Cs', 'Sr','Ba', 'Pb', 'Th', 'U']]
    Data_test_value=Data_test_value.fillna(0)#使用0填补空缺值
    #加载模型
    RF_RMI = pickle.load(open("RF_RMI.dat","rb"))
    RF_OC = pickle.load(open("RF_OC.dat","rb"))
    #应用分类器
    Predict_Test_RMI = RF_RMI.predict(Data_test_value)#预测RMI
    Predict_Test_OC = RF_OC.predict(Data_test_value)#预测OC
    Predict_Test_Type = 10*Predict_Test_RMI+Predict_Test_OC#综合结果
    Predict_Test_Type[Predict_Test_Type==1]=0 #将预测为MORA+Continental（1）的结果修正为MORA+Oceanic（0）
        
    #绘制预测结果柱状图
    Type = [0,10,20,11,21]
    Tectonic = ['MORA','OAA','OIA','CAA','CRA']
    Predict_Result = []
    for j in range(5):
        Count_Num = np.sum(Predict_Test_Type==Type[j])
        Predict_Result.append(Count_Num)
    Predict_Result = np.array(Predict_Result)
    import matplotlib.pyplot as plt
    tectonic = ['MORA','OAA','PRA','CAA','CRA']
    fig = plt.figure(figsize=(10,6))
    plt.title('Tectonic Environment Prediction',fontname='Times New Roman',fontsize=18)
    p = plt.bar(tectonic,100*Predict_Result/len(Data_test_value),color='silver',edgecolor='k',width=0.4)
    plt.bar_label(p,label_type='edge',fmt='%.2f',fontsize=14)
    plt.xticks(fontname='Times New Roman',fontsize=14)
    plt.yticks(fontname='Times New Roman',fontsize=14)
    plt.ylabel('Percentage(%)',fontname='Times New Roman',fontsize=14)
    plt.ylim(0,100)
    st.pyplot(fig)
    
    #0-MORA, 10-OAA, 20-OIA, 11-CAA, 21-CRA
    for i in range(5):
        Predict_Test_Type = pd.DataFrame(Predict_Test_Type)
        Predict_Test_Type.replace(Type[i],Tectonic[i],inplace=True)
    Predict_Test_Type = list(Predict_Test_Type.values)
    Data_test_all['Type']=Predict_Test_Type#将预测结果添加到原数据表格
    Data_test_all.to_csv('Data_predicted.csv',index=False)
    
    #创建下载按钮让用户下载含预测结果的CSV文件
    st.download_button(label='Download Predicted Data',
                      data= Data_test_all.to_csv(index=False),
                      file_name='Data_predicted.csv',
                      mime='text/csv')