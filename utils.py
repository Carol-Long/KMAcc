import numpy as np
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from helper_functions import MSCE as MSCE
from MAccWitness import MAccWitness
from itertools import product

# grid search on the best parameters
def grid_search_params(witness_metric, X_val, y_val):
    n_splits = 5
    kf = KFold(n_splits=n_splits, random_state = 42, shuffle=True)
    if witness_metric == 'rbf':
        gamma = np.arange(1, 10, 0.5)
        scores = np.zeros(len(gamma))
        for g in range(len(gamma)):
            wit = MAccWitness(gamma=gamma[g], metric=witness_metric)
            wit_model = make_pipeline(StandardScaler(), wit)
            score = cross_val_score(wit_model,X_val ,y_val,cv=kf,n_jobs=n_splits).mean()
            scores[g] = score
        #defining the optimal found witness function
        idx_max = np.nanargmax(scores.flatten())
        gopt = gamma[idx_max]
        print(f"Optimal Gamma: {gopt}")
        return gopt

    elif witness_metric == 'sigmoid':
        gamma = np.arange(1,10,0.5)
        coef0 = np.arange(-5, 5, 1)
        combos = list(product(range(len(gamma)), range(len(coef0))))
        scores = np.zeros((len(gamma), len(coef0)))
        #evaluating for each gamma the resulting witness function
        for g, c in combos:
            wit = MAccWitness(gamma=gamma[g], metric=witness_metric, coef0=coef0[c])
            wit_model = make_pipeline(StandardScaler(), wit)
            score = cross_val_score(wit_model,X_val,y_val,cv=kf,n_jobs=n_splits).mean()
            scores[g, c] = score
        #defining the optimal found witness function
        idx_max = np.nanargmax(scores.flatten())
        gopt, copt = combos[idx_max]
        gopt, copt = gamma[gopt], coef0[copt]
        print(f"Optimal Gamma: {gopt}, Optimal Coef: {copt}")
        return [gopt, copt]

    elif witness_metric == 'poly':
        # Optimal Gamma: 0.05, Optimal Coef: 1.5, Optimal Degree: 2
        gamma = [0.04, 0.05, 0.06, 0.1, 0.5]
        coef0 = [0, 0.5, 0.7, 1, 1.5, 2]
        degree = [2,3,4]
        combos = list(product(range(len(gamma)), range(len(coef0)), range(len(degree))))
        scores = np.zeros((len(gamma), len(coef0), len(degree)))

        #evaluating for each gamma the resulting witness function
        for g, c, d in combos:
            wit = MAccWitness(gamma=gamma[g], metric=witness_metric, coef0=coef0[c], degree=degree[d])
            wit_model = make_pipeline(StandardScaler(), wit)
            score = cross_val_score(wit_model,X_val,y_val,cv=kf,n_jobs=n_splits).mean()
            scores[g, c] = score
        #defining the optimal found witness function
        idx_max = np.nanargmax(scores.flatten())
        gopt, copt, dopt = combos[idx_max]
        gopt, copt, dopt = gamma[gopt], coef0[copt], degree[dopt]
        print(f"Optimal Gamma: {gopt}, Optimal Coef: {copt}, Optimal Degree: {dopt}")
        return [gopt, copt, dopt]
    
def gen_preds(model):
    return lambda x: model.predict_proba(x)[:, 1] # get predictions from MCBoost

def compute_KME(X_test, y_test, model, gopt, witness_metric):
  # K fold cross validation on the test set
  n_splits = 5
  kf = KFold(n_splits=n_splits, random_state = 42, shuffle=True)
  KME = np.zeros(n_splits)
  KME_std = np.zeros(n_splits)

  KME_witfit = np.zeros(n_splits)
  KME_witfit_std = np.zeros(n_splits)
  for i, (train_index, test_index) in enumerate(kf.split(X_test)):
    X_witfit, X_test_temp = X_test[train_index], X_test[test_index]
    y_witfit, y_test_temp = y_test[train_index], y_test[test_index]
    # initiate the witness model
    wit = MAccWitness(gamma=gopt, metric=witness_metric)
    wit_model = make_pipeline(StandardScaler(), wit)
    error_wit = y_witfit - model.predict_proba(X_witfit)[:, 1]
    wit_model.fit(X_witfit, error_wit)

    wit_witfit = wit_model.predict(X_witfit)
    yhat_proba_witfit = model.predict_proba(X_witfit)[:, 1]
    KME_witfit[i] = compute_calibration_error(wit_witfit, y_witfit, yhat_proba_witfit)
    KME_witfit_std[i] = np.std(wit_witfit * (y_witfit - yhat_proba_witfit))

    wit_test = wit_model.predict(X_test_temp)
    yhat_proba_test = model.predict_proba(X_test_temp)[:, 1]
    KME[i] = compute_calibration_error(wit_test, y_test_temp, yhat_proba_test)
    KME_std[i] = np.std(wit_test * (y_test_temp - yhat_proba_test))
  return KME.mean(), KME_std.mean(), KME_witfit.mean(), KME_witfit_std.mean()

# compute calibration error
def compute_calibration_error(wit_value, y_pred, yhat_proba):
  return np.abs((wit_value * (y_pred - yhat_proba)).mean())

def mega_compute_corrected(gopt, witness_metric, X_test, y_test, f_baseline, f_kMAcc, f_LSBoost, f_MCBoost, \
                           f_base_iso, f_kMAcc_iso, f_MCBoost_iso):

  # baseline KME, MSCE
  base_KME, base_KME_std, base_KME_witfit, base_KME_witfit_std = compute_KME(X_test, y_test, f_baseline, gopt, witness_metric)
  print(f"Baseline KME: {base_KME}, std {base_KME_std}")
  print(f"Baseline KME on witfit: {base_KME_witfit}, std {base_KME_witfit_std}")

  # KME for KMAcc
  kMAcc_KME, kMAcc_KME_std, kMAcc_KME_witfit, kMAcc_KME_witfit_std = compute_KME(X_test, y_test, f_kMAcc, gopt, witness_metric)
  print(f"KMAcc's KME: {kMAcc_KME}, std {kMAcc_KME_std}")
  print(f"KMAcc's KME on witfit: {kMAcc_KME_witfit}, std {kMAcc_KME_witfit_std}")

  # KME for LSBoost
  LSBoost_KME, LSBoost_KME_std, LSBoost_KME_witfit, LSBoost_KME_witfit_std = compute_KME(X_test, y_test, f_LSBoost, gopt, witness_metric)
  print(f"LSBoost's KME: {LSBoost_KME}, std {LSBoost_KME_std}")
  print(f"LSBoost's KME on witfit: {LSBoost_KME_witfit}, std {LSBoost_KME_witfit_std}")

  # KME for MCBoost
  MCBoost_KME, MCBoost_KME_std, MCBoost_KME_witfit, MCBoost_KME_witfit_std = compute_KME(X_test, y_test, f_MCBoost, gopt, witness_metric)
  print(f"MCBoost's KME: {MCBoost_KME}, std {MCBoost_KME_std}")
  print(f"MCBoost's KME on witfit: {MCBoost_KME_witfit}, std {MCBoost_KME_witfit_std}")

  # KME for baseline + isotonic
  base_iso_KME, base_iso_KME_std, base_iso_KME_witfit, base_iso_KME_witfit_std = compute_KME(X_test, y_test, f_base_iso, gopt, witness_metric)
  print(f"Baseline + isotonic's KME: {base_iso_KME}, std {base_iso_KME_std}")
  print(f"Baseline + isotonic's KME on witfit: {base_iso_KME_witfit}, std {base_iso_KME_witfit_std}")

  # KME for KMAcc + isotonic
  kMAcc_iso_KME, kMAcc_iso_KME_std, kMAcc_iso_KME_witfit, kMAcc_iso_KME_witfit_std = compute_KME(X_test, y_test, f_kMAcc_iso, gopt, witness_metric)
  print(f"KMAcc + isotonic's KME: {kMAcc_iso_KME}, std {kMAcc_iso_KME_std}")
  print(f"KMAcc + isotonic's KME on witfit: {kMAcc_iso_KME_witfit}, std {kMAcc_iso_KME_witfit_std}")

  # KME for MCBoost + isotonic
  MCBoost_iso_KME, MCBoost_iso_KME_std, MCBoost_iso_KME_witfit, MCBoost_iso_KME_witfit_std = compute_KME(X_test, y_test, f_MCBoost_iso, gopt, witness_metric)
  print(f"MCBoost + isotonic's KME: {MCBoost_iso_KME}, std {MCBoost_iso_KME_std}")
  print(f"MCBoost + isotonic's KME on witfit: {MCBoost_iso_KME_witfit}, std {MCBoost_iso_KME_witfit_std}")

  # predictions for the test set
  base_test = f_baseline.predict_proba(X_test)[:, 1]
  kMAcc_test = f_kMAcc.predict_proba(X_test)[:, 1]
  LSBoost_test = f_LSBoost.predict_proba(X_test)[:, 1]
  MCBoost_test = f_MCBoost.predict_proba(X_test)[:, 1]
  base_iso_test = f_base_iso.predict_proba(X_test)[:, 1]
  kMAcc_iso_test = f_kMAcc_iso.predict_proba(X_test)[:, 1]
  MCBoost_iso_test = f_MCBoost_iso.predict_proba(X_test)[:, 1]

  # compare standard calibration metric
  base_msce = MSCE(y_test, base_test)
  kMacc_msce = MSCE(y_test, kMAcc_test)
  LSBoost_msce = MSCE(y_test, LSBoost_test)
  MCBoost_msce = MSCE(y_test, MCBoost_test)
  base_iso_msce = MSCE(y_test, base_iso_test)
  kMacc_iso_msce = MSCE(y_test, kMAcc_iso_test)
  MCBoost_iso_msce = MSCE(y_test, MCBoost_iso_test)

  # compare AUC of the models
  AUC = [roc_auc_score(y_test, base_test), roc_auc_score(y_test, kMAcc_test), \
          roc_auc_score(y_test, LSBoost_test), roc_auc_score(y_test, MCBoost_test), \
          roc_auc_score(y_test, base_iso_test), roc_auc_score(y_test, kMAcc_iso_test), \
          roc_auc_score(y_test, MCBoost_iso_test)]

  return base_msce, base_KME, kMacc_msce, kMAcc_KME, LSBoost_msce, LSBoost_KME, \
          MCBoost_msce, MCBoost_KME, base_iso_msce, base_iso_KME, kMacc_iso_msce, kMAcc_iso_KME, \
          MCBoost_iso_msce, MCBoost_iso_KME, AUC

# def mega_compute_corrected(gopt, witness_metric, wit_test, X_wit, y_wit, X_test, y_test, wit_predictions_LS, test_predictions, yhat_proba_test,  g_wit, g_test, MCBoost_test, MCBoost_wit, \
#                            baseline_model, wit_prob_pos_isotonic, prob_pos_isotonic, wit_prob_pos_ablated, prob_pos_ablated, prob_test_mcboost_isotonic, prob_wit_mcboost_isotonic):

#   # ----------------------------------------
#   if baseline_model == "Logistic_Regression":
#     num_cluster = 5
#   elif baseline_model == "Decision_Tree":
#     num_cluster = 7
#   elif baseline_model == "Random_Forest":
#     num_cluster = 5
#   else:
#     num_cluster = 3

#   g_test_binned = KMeans(n_clusters=num_cluster, random_state=0, n_init="auto").fit(g_test.reshape(-1, 1))
#   g_wit_bin = KMeans(n_clusters=num_cluster, random_state=0, n_init="auto").fit(g_wit.reshape(-1, 1))
#   yhat_proba_test_binned = KMeans(n_clusters=num_cluster, random_state=0, n_init="auto").fit(yhat_proba_test.reshape(-1, 1))

#   g_new = np.zeros(len(g_test))
#   gwit_new = np.zeros(len(g_wit))
#   yhat_new = np.zeros(len(yhat_proba_test))
#   for i in range(num_cluster): # number of cluster here
#     ind1 = np.where(g_test_binned.labels_ ==i )
#     ind2 = np.where(yhat_proba_test_binned.labels_ == i)
#     ind3 = np.where(g_wit_bin.labels_==i)
#     g_new[ind1] = g_test_binned.cluster_centers_[i]
#     yhat_new[ind2] = yhat_proba_test_binned.cluster_centers_[i]
#     gwit_new[ind3] = g_wit_bin.cluster_centers_[i]

#   # print(g_test_binned_new)

#   # compare c* error_test and c* g_error_test
#   base_KCE = compute_calibration_error(wit_test, y_test, yhat_proba_test) # wit_test defined on f
  
#   KME_values_test = wit_test * (y_test - yhat_proba_test)
#   plt.hist(KME_values_test, bins=100)
#   plt.title(f'KME_test baseline, mean {np.mean(KME_values_test)} std {np.std(KME_values_test)} -- {baseline_model}')
#   plt.show()
#   #plot cdf
#   plt.hist(KME_values_test, bins=100, cumulative=True, density=True)
#   plt.title(f'CDF of KME_test baseline, mean {np.mean(KME_values_test)} std {np.std(KME_values_test)} -- {baseline_model}')
#   plt.show()
#   print(f"Baseline Kernel calibration error: {base_KCE}")

#   # define new witness on g (updated model)
#   wit = MAccWitness(gamma=gopt, metric=witness_metric)
#   wit_model = make_pipeline(StandardScaler(), wit)
#   error_wit = y_wit - g_wit
#   wit_model.fit(X_wit, error_wit)
#   wit_test_g = wit_model.predict(X_test)

#   kmulcal_KCE = compute_calibration_error(wit_test_g, y_test, g_test)
#   plt.hist(wit_test_g, bins=100)
#   plt.title(f'wit_test KMAcc, mean {np.mean(wit_test_g)} std {np.std(wit_test_g)} -- {baseline_model}')
#   plt.show()  
#   print(f"Our Method's kernel calibration error: {kmulcal_KCE}")

#   # define new witness on LSBoost (updated model)
#   error_wit = y_wit - wit_predictions_LS
#   wit_model.fit(X_wit, error_wit)
#   wit_test_LS = wit_model.predict(X_test)
#   lsboost_KCE = compute_calibration_error(wit_test_LS,y_test, test_predictions)
#   plt.hist(wit_test_LS, bins=100)
#   plt.title(f'wit_test LSBoost, mean {np.mean(wit_test_LS)} std {np.std(wit_test_LS)} -- {baseline_model}')
#   plt.show()  
#   print(f"LSBoost kernel calibration error: {lsboost_KCE}")

#   # # kernel error for binned f(x) and g(x)
#   # # define new witness on binned baseline(updated model)
#   # val_binned = KMeans(n_clusters=num_cluster, random_state=0, n_init="auto").fit(y_wit.reshape(-1, 1))
#   # error_wit = y_wit - val_binned
#   # wit_model.fit(X_wit, error_wit)
#   # wit_test_f_binned = wit_model.predict(X_test)
#   # base_binned_KCE = compute_calibration_error(wit_test_f_binned, y_test, yhat_new)
#   # print(f"Baseline Kernel calibration error, binned: {base_binned_KCE}")
#   base_binned_KCE  = 0 #placeholder

#   # define new witness on binned g(updated model)
#   error_wit = y_wit - gwit_new
#   wit_model.fit(X_wit, error_wit)
#   wit_test_g_binned = wit_model.predict(X_test)
#   kmulcal_binned_KCE = compute_calibration_error(wit_test_g_binned, y_test, g_new)
#   #print(f"Our Method's Kernel calibration error, binned: {kmulcal_binned_KCE}")

#   # compare standard calibration metric
#   base_msce = MSCE(y_test, yhat_proba_test)
#   #print(f'Baseline MSCE: {base_msce:.6f}')

#   kmulcal_msce = MSCE(y_test, g_test)
#   #print(f'Our Method\'s MSCE: {kmulcal_msce:.6f}')

#   lsboost_msce = MSCE(y_test, test_predictions)
#   # print(f'Our Method\'s MS Calibration Error: {MSCE(y_test, g_test):.6f}')
#   #print(f'LSBoost MSCE: {lsboost_msce:.6f}')

#   # MSCE for binned f(x) and g(x)
#   base_msce_binned = MSCE(y_test, yhat_new)
#   #print(f'Baseline MSCE, binned: {base_msce_binned:.6f}')

#   kmulcal_msce_binned = MSCE(y_test, g_new)
#   #print(f'Our Method\'s MSCE, binned: {kmulcal_msce_binned:.6f}')

#   MCBoost_msce = MSCE(y_test, MCBoost_test)
#   # define new witness on MCBoost(updated model)
#   error_wit = y_wit - MCBoost_wit
#   wit_model.fit(X_wit, error_wit)
#   wit_test_mcboost = wit_model.predict(X_test)
#   MCBoost_KCE = compute_calibration_error(wit_test_mcboost, y_test, MCBoost_test)

#   wit_witset_mcboost = wit_model.predict(X_wit)
#   KME_values_wit = wit_witset_mcboost * error_wit
#   plt.hist(KME_values_wit, bins=100)
#   plt.title(f'KME_witset MCBoost, mean {np.mean(KME_values_wit)} std {np.std(KME_values_wit)} -- {baseline_model}')
#   plt.show()
#   #plot cdf
#   plt.hist(KME_values_wit, bins=100, cumulative=True, density=True)
#   plt.title(f'CDF of KME_witset MCBoost, mean {np.mean(KME_values_wit)} std {np.std(KME_values_wit)} -- {baseline_model}')
#   plt.show()

#   KME_values_test = wit_test_mcboost * (y_test - MCBoost_test)
#   plt.hist(KME_values_test, bins=100)
#   plt.title(f'KME_test MCBoost, mean {np.mean(KME_values_test)} std {np.std(KME_values_test)} -- {baseline_model}')
#   plt.show()
#   #plot cdf
#   plt.hist(KME_values_test, bins=100, cumulative=True, density=True)
#   plt.title(f'CDF of KME_test MCBoost, mean {np.mean(KME_values_test)} std {np.std(KME_values_test)} -- {baseline_model}')
#   plt.show()

#   print(f"MCBoost's Kernel calibration error, binned: {MCBoost_KCE}")

#   MCBoost_iso_msce = MSCE(y_test, prob_test_mcboost_isotonic)
#   # define new witness on MCBoost + isotonic
#   error_wit = y_wit - prob_wit_mcboost_isotonic
#   wit_model.fit(X_wit, error_wit)
#   wit_test_mcboost_iso = wit_model.predict(X_test)
#   MCBoost_iso_KCE = compute_calibration_error(wit_test_mcboost_iso, y_test, prob_test_mcboost_isotonic)

#   isotonic_cal_msce = MSCE(y_test, prob_pos_isotonic)
#   # define new witness on KMAcc+isotonic (updated model)
#   error_wit = y_wit - wit_prob_pos_isotonic
#   wit_model.fit(X_wit, error_wit)
#   wit_test_iso = wit_model.predict(X_test)
#   isotonic_cal_KCE = compute_calibration_error(wit_test_iso, y_test, prob_pos_isotonic)

#   ablated_msce = MSCE(y_test, prob_pos_ablated)
#   # define new witness on baseline+isotonic (updated model)
#   error_wit = y_wit - wit_prob_pos_ablated
#   wit_model.fit(X_wit, error_wit)
#   wit_test_ablated = wit_model.predict(X_test)
#   ablated_KCE = compute_calibration_error(wit_test_ablated, y_test, prob_pos_ablated)

#   # auc = roc_auc_score(y_test, )
#   AUC = [roc_auc_score(y_test, yhat_proba_test), roc_auc_score(y_test, g_test), \
#          roc_auc_score(y_test, test_predictions), \
#          roc_auc_score(y_test, g_new), roc_auc_score(y_test, MCBoost_test), \
#          roc_auc_score(y_test, prob_pos_isotonic), \
#          roc_auc_score(y_test, prob_pos_ablated),\
#          roc_auc_score(y_test, prob_test_mcboost_isotonic)] #,


#   return base_msce, base_KCE, kmulcal_msce, kmulcal_KCE, lsboost_msce, lsboost_KCE, \
#          base_msce_binned, base_binned_KCE, kmulcal_msce_binned, kmulcal_binned_KCE, \
#          MCBoost_msce, MCBoost_KCE, isotonic_cal_msce, isotonic_cal_KCE, \
#          ablated_msce, ablated_KCE, AUC, MCBoost_iso_msce, MCBoost_iso_KCE



# def mega_compute(wit_test, y_test, yhat_proba_test, test_predictions, g_test, MCBoost_test, baseline_model, prob_pos_isotonic, prob_pos_sigmoid, prob_pos_ablated):

#   if baseline_model == "Logistic_Regression":
#     num_cluster = 5
#   elif baseline_model == "Decision_Tree":
#     num_cluster = 7
#   elif baseline_model == "Random_Forest":
#     num_cluster = 5
#   else:
#     num_cluster = 3

#   g_test_binned = KMeans(n_clusters=num_cluster, random_state=0, n_init="auto").fit(g_test.reshape(-1, 1))

#   yhat_proba_test_binned = KMeans(n_clusters=num_cluster, random_state=0, n_init="auto").fit(yhat_proba_test.reshape(-1, 1))

#   g_new = np.zeros(len(g_test))
#   yhat_new = np.zeros(len(yhat_proba_test))
#   for i in range(num_cluster): # number of cluster here
#     ind1 = np.where(g_test_binned.labels_ ==i )
#     ind2 = np.where(yhat_proba_test_binned.labels_ == i)
#     g_new[ind1] = g_test_binned.cluster_centers_[i]
#     yhat_new[ind2] = yhat_proba_test_binned.cluster_centers_[i]

#   # print(g_test_binned_new)

#   # compare c* error_test and c* g_error_test
#   base_KCE = compute_calibration_error(wit_test, y_test, yhat_proba_test)
#   print(f"Baseline Kernel calibration error: {base_KCE}")

#   kmulcal_KCE = compute_calibration_error(wit_test, y_test, g_test)
#   print(f"Our Method's kernel calibration error: {kmulcal_KCE}")

#   lsboost_KCE = compute_calibration_error(wit_test,y_test, test_predictions)
#   print(f"LSBoost kernel calibration error: {lsboost_KCE}")

#   # kernel error for binned f(x) and g(x)
#   base_binned_KCE = compute_calibration_error(wit_test, y_test, yhat_new)
#   print(f"Baseline Kernel calibration error, binned: {base_binned_KCE}")

#   kmulcal_binned_KCE = compute_calibration_error(wit_test, y_test, g_new)
#   print(f"Our Method's Kernel calibration error, binned: {kmulcal_binned_KCE}")

#   # # condition calibration error
#   # con_cal_error1 = con_cal_err(num_bins, wit_test, y_test, g_test)
#   # print(f"our condition calibration error: {con_cal_error1}")

#   # con_cal_error2 = con_cal_err(num_bins, wit_test, y_test, test_predictions)
#   # print(f"BoostReg condition calibration error: {con_cal_error2}")

#   # compare standard calibration metric
#   base_msce = MSCE(y_test, yhat_proba_test)
#   print(f'Baseline MSCE: {base_msce:.6f}')

#   kmulcal_msce = MSCE(y_test, g_test)
#   print(f'Our Method\'s MSCE: {kmulcal_msce:.6f}')

#   lsboost_msce = MSCE(y_test, test_predictions)
#   # print(f'Our Method\'s MS Calibration Error: {MSCE(y_test, g_test):.6f}')
#   print(f'LSBoost MSCE: {lsboost_msce:.6f}')

#   # MSCE for binned f(x) and g(x)
#   base_msce_binned = MSCE(y_test, yhat_new)
#   print(f'Baseline MSCE, binned: {base_msce_binned:.6f}')

#   kmulcal_msce_binned = MSCE(y_test, g_new)
#   print(f'Our Method\'s MSCE, binned: {kmulcal_msce_binned:.6f}')

#   MCBoost_msce = MSCE(y_test, MCBoost_test)
#   MCBoost_KCE = compute_calibration_error(wit_test, y_test, MCBoost_test)

#   isotonic_cal_msce = MSCE(y_test, prob_pos_isotonic)
#   isotonic_cal_KCE = compute_calibration_error(wit_test, y_test, prob_pos_isotonic)

#   sigmoid_cal_msce = MSCE(y_test, prob_pos_sigmoid)
#   sigmoid_cal_KCE = compute_calibration_error(wit_test, y_test, prob_pos_sigmoid)

#   ablated_msce = MSCE(y_test, prob_pos_ablated)
#   ablated_KCE = compute_calibration_error(wit_test, y_test, prob_pos_ablated)

#   import matplotlib.pyplot as plt
#   plt.hist(g_new, label='ours')
#   plt.hist(test_predictions)
#   plt.title(f'K Means data visualization for {baseline_model}')
#   plt.show()

#   # auc = roc_auc_score(y_test, )
#   AUC = [roc_auc_score(y_test, yhat_proba_test), roc_auc_score(y_test, g_test), \
#          roc_auc_score(y_test, test_predictions), \
#          roc_auc_score(y_test, g_new), roc_auc_score(y_test, MCBoost_test), \
#          roc_auc_score(y_test, prob_pos_isotonic), \
#          roc_auc_score(y_test, prob_pos_ablated)] #,


#   return base_msce, base_KCE, kmulcal_msce, kmulcal_KCE, lsboost_msce, lsboost_KCE, \
#          base_msce_binned, base_binned_KCE, kmulcal_msce_binned, kmulcal_binned_KCE, \
#          MCBoost_msce, MCBoost_KCE, isotonic_cal_msce, isotonic_cal_KCE, sigmoid_cal_msce,\
#          sigmoid_cal_KCE, ablated_msce, ablated_KCE, AUC


# # conditioned calibration error
# def con_cal_err(bin, wit_value, y_pred, yhat_proba):
#   '''
#   bin is the number of slices
#   wit_value is c*(x)
#   y_pred is the predicted label (0 or 1)
#   yhat_proba is the probability score (between 0 and 1)
#   '''
#   err = 0
#   incre = 1/bin
#   for i in range(bin):
#     ind = np.where(np.logical_and(yhat_proba>i*incre, yhat_proba<=(i+1)*incre))
#     temp_sum = compute_calibration_error(wit_value[ind], y_pred[ind], yhat_proba[ind])
#     # print(temp_sum)
#     if (not np.isnan(temp_sum)):
#       frac = len(ind[0]) / len(yhat_proba)
#       err += (temp_sum ** 2) * frac
#   return err