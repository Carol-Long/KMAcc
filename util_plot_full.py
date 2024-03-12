#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:53:14 2024

@author: carollong
"""

import pandas as pd
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from util_plot import plotter, plot_in_row


# one plot for each dataset
sns.set(style="whitegrid", color_codes=True)  
base_classifiers = ["Logistic_Regression", "Naive_Bayes", "Random_Forest", "NN"]

prefixs= ["EMP_MA_", "EMP_AL_", "Income_WA_", "Income_IL_", \
          "Health_WI_", "Health_OH_", "Mobility_NJ_", "Mobility_NY_"]
    
for prefix in prefixs:
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



# Figure 2
sns.set(style="whitegrid", color_codes=True)
retrieve_df1 = pd.read_csv("Income_WA_all.csv")
retrieve_df1.drop(columns=retrieve_df1.columns[0], axis=1,  inplace=True)
data_class_retrieved1 = retrieve_df1.values.tolist()
for i in range(len(data_class_retrieved1)):
  for j in range(len(data_class_retrieved1[i])):
    data_class_retrieved1[i][j] = ast.literal_eval(data_class_retrieved1[i][j])
data_class = data_class_retrieved1

retrieve_df2 = pd.read_csv("Health_WI_all.csv")
retrieve_df2.drop(columns=retrieve_df2.columns[0], axis=1,  inplace=True)
data_class_retrieved2 = retrieve_df2.values.tolist()
for i in range(len(data_class_retrieved2)):
  for j in range(len(data_class_retrieved2[i])):
    data_class_retrieved2[i][j] = ast.literal_eval(data_class_retrieved2[i][j])
data2 = data_class_retrieved2

base_classifiers = ["Logistic_Regression", "Naive_Bayes", "Random_Forest", "NN"]

fig, ax = plt.subplots(nrows=2, ncols=4)

plt.rcParams["font.family"] = "monospace"
plt.rcParams['font.size']=15

fig.set_size_inches((12, 6))

loc_data = [[(-0.01, 0.006), (-0.015, 0.002), (0, 0.001), (.025, .01), (0, -0.025), (0, -.02), (0.15, 0.0), (0,0.02)], \
            [(0, -.02), (.15, .005), (-.02, .003), (.15, .008), (.1, -.02), (0.0, -.02), (0.2, 0.0), (0,0.02)], \
            [(-.05, .003), (.06, .005), (0, .002), (-.04, .001), (0, -.02), (.0, -.02), (.13, -.005), (0,0.02)], \
            [(.0015, -.02), (0, .005), (0, .002), (.005, .01), (.05, -0.007), (.00, -.02), (0.08, -0.005), (0,0.02)]]



loc_data_2 = [[(0.01, 0.006), (0.005, 0.002), (0, 0.001), (-.03, .008), (0, -0.02), (0, -.023), (0, 0.005), (0,0.02)], \
            [(0, -0.025), (0, 0.003), (0, 0.002), (0, 0.005), (0, 0), (0, -.02), (-0.01, -.02), (-0.03,0.0)], \
            [(0.01, 0.006), (0, .005), (0, .002), (0, .004), (0, -0.025), (0, -.02), (0.06, -.005), (0,0.018)], \
            [(0.013, -0.01), (0, .005), (0, .002), (.02, -.013), (-.002, -0.025), (0, -.02), (0, .003), (0,0.02)]]


loc_data_nil = [[(0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00)], \
            [(0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00)], \
            [(0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00)], \
            [(0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00), (0.0, 0.00)]]
        
base, kma, lsb, kmak, mcb, iso, abl, mcb_iso = plot_in_row(ax, 0, data_class, loc_data)
plot_in_row(ax, 1, data2, loc_data_2)

"""
for col in ax[1].reshape(-1):
  col.axis("off")"""

fig.tight_layout()
#fig.legend((base, kma, lsb, kmak, mcb, iso, abl), (f'baseline', f'KMAcc', f'LS Boost', f'KMAcc + KMeans', f'MC Boost', f'KMAcc + Isotonic Calibration', f'Baseline + Isotonic Calibration'), loc='outside lower center', ncols = 4)
fig.legend((base, kma, lsb, kmak, mcb, iso, abl, mcb_iso), ('baseline', 'KMAcc', 'LS Boost', 'KMAcc + KMeans', \
                                                            'MC Boost', 'KMAcc + Isotonic Calibration', 'Baseline + Isotonic Calibration', 'MC Boost + Isotonic Calibration'),bbox_to_anchor=(0.5, -0.05),loc='center', ncols = 4)
plt.savefig("Fig2_KMEcorrected.png",bbox_inches='tight',dpi=300)
plt.show()







