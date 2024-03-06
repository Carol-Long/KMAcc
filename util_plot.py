#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:45:14 2024

@author: carollong
"""
import pandas as pd
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_in_row(ax, row, data_class, loc_data):
    i = 0
    for col in ax[row].reshape(-1):
      classifier = base_classifiers[i]
      base_msce, base_KCE, kmulcal_msce, kmulcal_KCE, lsboost_msce, lsboost_KCE, base_msce_binned, base_binned_KCE, \
      kmulcal_msce_binned, kmulcal_binned_KCE, MCBoost_msce, MCBoost_KCE, isotonic_msce, \
      isotonic_KCE, sigmoid_msce, sigmoid_KCE, ablated_msce, ablated_KCE, AUC, MCBoost_iso_msce, MCBoost_iso_KCE  = data_class[i]
      
      x = np.array([np.mean(base_KCE, axis = 0), np.mean(kmulcal_KCE, axis = 0), np.mean(lsboost_KCE, axis = 0), \
                    np.mean(kmulcal_binned_KCE, axis = 0), np.mean(MCBoost_KCE, axis = 0), \
                    np.mean(isotonic_KCE, axis = 0), np.mean(ablated_KCE, axis = 0), np.mean(MCBoost_iso_KCE, axis = 0)])
          
      y = np.array([np.mean(base_msce, axis = 0), np.mean(kmulcal_msce, axis = 0), np.mean(lsboost_msce, axis = 0), \
                    np.mean(kmulcal_msce_binned, axis = 0), np.mean(MCBoost_msce, axis=0), \
                    np.mean(isotonic_msce, axis = 0), np.mean(ablated_msce, axis = 0), np.mean(MCBoost_iso_msce, axis = 0)])
      z = np.mean(AUC, axis = 0)
      x_std = np.array([np.std(base_KCE, axis = 0), np.std(kmulcal_KCE, axis = 0), np.std(lsboost_KCE, axis = 0), \
                        np.std(kmulcal_binned_KCE, axis = 0), np.std(MCBoost_KCE, axis = 0), \
                        np.std(isotonic_KCE, axis = 0), np.std(ablated_KCE, axis = 0), np.std(MCBoost_iso_KCE, axis = 0)])
      y_std = np.array([np.std(base_msce, axis = 0), np.std(kmulcal_msce, axis = 0), np.std(lsboost_msce, axis = 0),\
                        np.std(kmulcal_msce_binned, axis = 0), np.std(MCBoost_msce, axis=0), \
                        np.std(isotonic_msce, axis = 0), np.std(ablated_msce, axis = 0), np.std(MCBoost_iso_msce, axis = 0)])
    
      colors = ['steelblue','darkgoldenrod','coral','gray','limegreen', 'darkred', 'indigo', 'green']
      markersize = 40
      #sns.set(style="whitegrid", color_codes=True)
      base = col.scatter(x[0], y[0], c=colors[0], marker = 'o', s = markersize)
      kma = col.scatter(x[1], y[1], c=colors[1], marker = 's', s = markersize)
      lsb = col.scatter(x[2], y[2], c=colors[2], marker = 'p', s = markersize)
      kmak = col.scatter(x[3], y[3], c=colors[3], marker = 'x', s = markersize)
      mcb = col.scatter(x[4], y[4], c=colors[4], marker = '^', s = markersize)
      iso = col.scatter(x[5], y[5], c=colors[5], marker = 'd', s = markersize)
      abl = col.scatter(x[6], y[6], c=colors[6], marker = 'h', s = markersize)
      mcb_iso = col.scatter(x[7], y[7], c=colors[7], marker = '<', s = markersize)  
    
      col.errorbar(x[0], y[0], xerr=x_std[0], yerr=y_std[0], linestyle='', color = colors[0], alpha = 1)
      col.errorbar(x[1], y[1], xerr=x_std[1], yerr=y_std[1], linestyle='', color= colors[1], alpha = 1)
      col.errorbar(x[2], y[2], xerr=x_std[2], yerr=y_std[2], linestyle='', color= colors[2], alpha = 1)
      col.errorbar(x[3], y[3], xerr=x_std[3], yerr=y_std[3], linestyle='', color= colors[3], alpha = 1)
      col.errorbar(x[4], y[4], xerr=x_std[4], yerr=y_std[4], linestyle='', color= colors[4], alpha = 1)
      col.errorbar(x[5], y[5], xerr=x_std[5], yerr=y_std[5], linestyle='', color= colors[5], alpha = 1)
      col.errorbar(x[6], y[6], xerr=x_std[6], yerr=y_std[6], linestyle='', color= colors[6], alpha = 1)
      col.errorbar(x[7], y[7], xerr=x_std[7], yerr=y_std[7], linestyle='', color= colors[7], alpha = 1)
    
      col.set_xlabel('KME', weight='bold')
      col.set_ylabel('MSCE', weight='bold')
    
    
      cl = classifier.split("_")
      if len(cl) == 1:
        cl = cl[0]
      else:
        cl = cl[0] + " " + cl[1]
    
      col.set_title(cl, weight='bold')
      col.tick_params(axis='both', labelsize=9)
      col.set_ylim(bottom=-.01, top=.2)
      col.set_xlim(left=0)
    
    
      l = [f'{score:.3f}' for score in z]  # Label AUC for each point
      #texts = [col.text(x[j],y[j],label, fontsize = 9, ha='center', va='bottom', c = colors[j]) for j, label in enumerate(l)]
      # adjust_text(texts)
      #ta.allocate_text(fig,ax[row],x,y,l, textsize=9, draw_lines= False)
      
      for j, label in enumerate(l):
        col.text(x[j] + loc_data[i][j][0], y[j] + loc_data[i][j][1], label, fontsize=9, ha='center', va='bottom', c = colors[j])
      
      i += 1
    return base, kma, lsb, kmak, mcb, iso, abl, mcb_iso
  
    
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















