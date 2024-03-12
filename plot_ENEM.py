#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 08:26:32 2024

@author: carollong
"""


from util_plot import plotter
import seaborn as sns
import pandas as pd
import ast
import matplotlib.pyplot as plt

sns.set(style="whitegrid", color_codes=True)  
base_classifiers = ["Logistic_Regression", "Naive_Bayes", "Random_Forest", "NN"]


prefix = "ENEM_"
    

csvfile = prefix +"all.csv"
retrieve_df1 = pd.read_csv(csvfile)
retrieve_df1.drop(columns=retrieve_df1.columns[0], axis=1,  inplace=True)
data_class = retrieve_df1.values.tolist()
for i in range(len(data_class)):
  for j in range(len(data_class[i])):
    data_class[i][j] = ast.literal_eval(data_class[i][j])


fig, axs = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches((12, 8))
plt.rcParams["font.family"] = "monospace"
plt.rcParams['font.size']=15


for i in range(len(base_classifiers)):
    base, kma, lsb, kmak, mcb, iso, abl, mcb_iso = plotter(axs[i//2,i%2], data_class[i], base_classifiers[i],prefix)
    
fig.tight_layout()    
fig.legend((base, kma, lsb, kmak, mcb, iso, abl, mcb_iso), ('baseline', 'KMAcc', 'LS Boost', 'KMAcc + KMeans', \
                                                            'MC Boost', 'KMAcc + Isotonic Calibration', 'Baseline + Isotonic Calibration', 'MC Boost + Isotonic Calibration'),bbox_to_anchor=(0.5, -0.05),loc='center', ncols = 4)


plt.savefig('plots/'+ prefix  + 'all.png',bbox_inches='tight',dpi=300,format='png')
plt.show()

