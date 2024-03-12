#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:45:14 2024

@author: carollong
"""
import numpy as np
from adjustText import adjust_text

def plotter(ax, data_class, baseline_model, prefix):
    base_msce, base_KCE, kmulcal_msce, kmulcal_KCE, lsboost_msce, lsboost_KCE, base_msce_binned, base_binned_KCE, kmulcal_msce_binned, kmulcal_binned_KCE, \
        MCBoost_msce, MCBoost_KCE, isotonic_msce, isotonic_KCE, sigmoid_msce, sigmoid_KCE, ablated_msce, ablated_KCE, AUC, MCBoost_iso_msce, MCBoost_iso_KCE = data_class
    x = np.array([np.mean(base_KCE, axis = 0), np.mean(kmulcal_KCE, axis = 0), np.mean(lsboost_KCE, axis = 0), \
                np.mean(kmulcal_binned_KCE, axis = 0), np.mean(MCBoost_KCE, axis = 0), \
                np.mean(isotonic_KCE, axis = 0), np.mean(ablated_KCE, axis = 0), np.mean(MCBoost_iso_KCE, axis = 0)])
      
    y = np.array([np.mean(base_msce, axis = 0), np.mean(kmulcal_msce, axis = 0), np.mean(lsboost_msce, axis = 0), \
                np.mean(kmulcal_msce_binned, axis = 0), np.mean(MCBoost_msce, axis=0), \
                np.mean(isotonic_msce, axis = 0), np.mean(ablated_msce, axis = 0), np.mean(MCBoost_iso_msce, axis = 0)])
    z = np.mean(AUC, axis = 0) ## replace with AUC values
    # l = ['Baseline', 'KMultiCal', 'LS Boost', 'Baseline + KMeans', 'KMultiCal + KMeans']
    
    # markers = ['o', 's', '^', 'D', 'v']
    
    x_std = np.array([np.std(base_KCE, axis = 0), np.std(kmulcal_KCE, axis = 0), np.std(lsboost_KCE, axis = 0), \
                    np.std(kmulcal_binned_KCE, axis = 0), np.std(MCBoost_KCE, axis = 0), \
                    np.std(isotonic_KCE, axis = 0), np.std(ablated_KCE, axis = 0), np.std(MCBoost_iso_KCE, axis = 0)])
    y_std = np.array([np.std(base_msce, axis = 0), np.std(kmulcal_msce, axis = 0), np.std(lsboost_msce, axis = 0),\
                    np.std(kmulcal_msce_binned, axis = 0), np.std(MCBoost_msce, axis=0), \
                    np.std(isotonic_msce, axis = 0), np.std(ablated_msce, axis = 0), np.std(MCBoost_iso_msce, axis = 0)])

    colors = ['steelblue','darkgoldenrod','coral','gray','limegreen', 'darkred', 'indigo', 'green']
    markersize = 40
    base = ax.scatter(x[0], y[0], c=colors[0], marker = 'o', s = markersize)
    kma = ax.scatter(x[1], y[1], c=colors[1], marker = 's', s = markersize)
    lsb = ax.scatter(x[2], y[2], c=colors[2], marker = 'p', s = markersize)
    kmak = ax.scatter(x[3], y[3], c=colors[3], marker = 'x', s = markersize)
    mcb = ax.scatter(x[4], y[4], c=colors[4], marker = '^', s = markersize)
    iso = ax.scatter(x[5], y[5], c=colors[5], marker = 'd', s = markersize)
    abl = ax.scatter(x[6], y[6], c=colors[6], marker = 'h', s = markersize)
    mcb_iso = ax.scatter(x[7], y[7], c=colors[7], marker = '<', s = markersize)  
      
    ax.errorbar(x[0], y[0], xerr=x_std[0], yerr=y_std[0], linestyle='', color = colors[0], alpha = 1)
    ax.errorbar(x[1], y[1], xerr=x_std[1], yerr=y_std[1], linestyle='', color= colors[1], alpha = 1)
    ax.errorbar(x[2], y[2], xerr=x_std[2], yerr=y_std[2], linestyle='', color= colors[2], alpha = 1)
    ax.errorbar(x[3], y[3], xerr=x_std[3], yerr=y_std[3], linestyle='', color= colors[3], alpha = 1)
    ax.errorbar(x[4], y[4], xerr=x_std[4], yerr=y_std[4], linestyle='', color= colors[4], alpha = 1)
    ax.errorbar(x[5], y[5], xerr=x_std[5], yerr=y_std[5], linestyle='', color= colors[5], alpha = 1)
    ax.errorbar(x[6], y[6], xerr=x_std[6], yerr=y_std[6], linestyle='', color= colors[6], alpha = 1)
    ax.errorbar(x[7], y[7], xerr=x_std[7], yerr=y_std[7], linestyle='', color= colors[7], alpha = 1)

    '''
    norm = plt.Normalize(0.7, 1)
    m = plt.cm.ScalarMappable(cmap="Reds", norm = norm)
    m.set_array([])
    '''
    # cm=plt.get_cmap('Reds')
    # for i, (xval, yval, x_error_val, y_error_val, zval) in enumerate(zip(x, y, x_std, y_std, z)):
    #     # colour=cm(0.7*zval)
    #     # print(zval)
    #     plt.errorbar(xval, yval, xerr=x_error_val, yerr=y_error_val, linestyle='', ecolor=cm(zval), alpha = 0.3, capsize = 3)
    
    # base.figure.colorbar(m, label = 'AUC')
    # cbar = plt.colorbar(base)
    # cbar.set_label('AUC')
    
    ax.set_xlabel('KME', weight='bold')
    ax.set_ylabel('MSCE', weight='bold')
    
    if baseline_model == "Logistic_Regression":
      ax.set_title('Logistic Regression', weight='bold')
    elif baseline_model == "Decision_Tree":
      ax.set_title('Decision Tree', weight='bold')
    elif baseline_model == "Random_Forest":
      ax.set_title('Random Forest', weight='bold')
    elif baseline_model == "Kernel_SVM":
      ax.set_title('Kernel SVM', weight='bold')
    elif baseline_model == "Naive_Bayes":
      ax.set_title("Gaussian Naive Bayes", weight='bold')
    elif baseline_model == "NN":
      ax.set_title("Neural Network", weight='bold')
    else:
      ax.set_title('')

    l = [f'{score:.3f}' for score in z]  # Label AUC for each point
    texts = [ax.text(x[j],y[j],label, fontsize = 10, ha='center', va='bottom', c = colors[j]) for j, label in enumerate(l)]
    adjust_text(texts, ax=ax, force_static = (0.5,1))
    #adjust_text(texts, ax=ax)
    ax.tick_params(axis='both', labelsize=9)
    ax.set_ylim(bottom=-.01, top=.28)
    ax.set_xlim(left=0)
    
    return base, kma, lsb, kmak, mcb, iso, abl, mcb_iso


def plot_in_row(ax, row, data_class, loc_data):
    i = 0
    base_classifiers = ["Logistic_Regression", "Naive_Bayes", "Random_Forest", "NN"]
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
  
    












