import numpy as np
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from MAccWitness import MAccWitness
from itertools import product

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
    yhat_proba_witfit = model.predict_proba(X_witfit)[:, 1]    
    error_wit = y_witfit - yhat_proba_witfit
    wit_model.fit(X_witfit, error_wit)

    wit_witfit = wit_model.predict(X_witfit)
    KME_witfit[i] = wit.compute_KME(X_witfit, y_witfit, yhat_proba_witfit) 
    KME_witfit_std[i] = np.std(wit_witfit * (y_witfit - yhat_proba_witfit))

    wit_test = wit_model.predict(X_test_temp)
    yhat_proba_test = model.predict_proba(X_test_temp)[:, 1]
    KME[i] = wit.compute_KME(X_test_temp, y_test_temp, yhat_proba_test) 
    KME_std[i] = np.std(wit_test * (y_test_temp - yhat_proba_test))
  return KME.mean(), KME_std.mean(), KME_witfit.mean(), KME_witfit_std.mean()

def MSCE_new(y_test, y_proba, num_bins=20):
    # get the difference
    difference_array = y_test - y_proba
    
    # initialize bins
    bin_width = 1 / num_bins
    residuals_binned = np.zeros(num_bins)
    count_per_bin = np.zeros(num_bins)
    
    for i in range(len(y_proba)):
        bin_index = int(y_proba[i] // bin_width)
        residuals_binned[bin_index] += difference_array[i]
        count_per_bin[bin_index] += 1
    
    # avoid divide by 0
    count_per_bin[count_per_bin == 0] = 1
    # print(residuals_binned)
    # print(count_per_bin)
    residuals_binned = (residuals_binned/count_per_bin)**2 
    msce_value = np.sum(residuals_binned* (count_per_bin/ len(y_proba)))
    # print(residuals_binned)
    return msce_value

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
  base_msce = MSCE_new(y_test, base_test)
  kMacc_msce = MSCE_new(y_test, kMAcc_test)
  LSBoost_msce = MSCE_new(y_test, LSBoost_test)
  MCBoost_msce = MSCE_new(y_test, MCBoost_test)
  base_iso_msce = MSCE_new(y_test, base_iso_test)
  kMacc_iso_msce = MSCE_new(y_test, kMAcc_iso_test)
  MCBoost_iso_msce = MSCE_new(y_test, MCBoost_iso_test)

  # compare KME of the models
  KMEs = [base_KME, kMAcc_KME, LSBoost_KME, MCBoost_KME, base_iso_KME, kMAcc_iso_KME, MCBoost_iso_KME]

  # compare MSCE (calibration metric) of the models
  MSCEs = [base_msce, kMacc_msce, LSBoost_msce, MCBoost_msce, base_iso_msce, kMacc_iso_msce, MCBoost_iso_msce]

  # compare AUC of the models
  AUCs = [roc_auc_score(y_test, base_test), roc_auc_score(y_test, kMAcc_test), \
          roc_auc_score(y_test, LSBoost_test), roc_auc_score(y_test, MCBoost_test), \
          roc_auc_score(y_test, base_iso_test), roc_auc_score(y_test, kMAcc_iso_test), \
          roc_auc_score(y_test, MCBoost_iso_test)]

  # compare classification error of the models
  Classification_Errors = [np.mean((base_test > 0.5) ^ y_test), np.mean((kMAcc_test > 0.5) ^ y_test), \
          np.mean((LSBoost_test > 0.5) ^ y_test), np.mean((MCBoost_test > 0.5) ^ y_test), \
          np.mean((base_iso_test > 0.5) ^ y_test), np.mean((kMAcc_iso_test > 0.5) ^ y_test), \
          np.mean((MCBoost_iso_test > 0.5) ^ y_test)]
  
  # compare Mean Squared Error of the models
  MSEs = [np.mean((base_test - y_test)**2), np.mean((kMAcc_test - y_test)**2), \
          np.mean((LSBoost_test - y_test)**2), np.mean((MCBoost_test - y_test)**2), \
          np.mean((base_iso_test - y_test)**2), np.mean((kMAcc_iso_test - y_test)**2), \
          np.mean((MCBoost_iso_test - y_test)**2)]
  
  return KMEs, MSCEs, AUCs, Classification_Errors, MSEs

# grid search on the best parameters
def grid_search_params(witness_metric, X_val, y_val):
    #return 1 # reduce complexity of the model
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
    