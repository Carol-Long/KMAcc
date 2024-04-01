import sys
sys.path.append('Level-Set-Boosting')
from utils import *
from KMultiAcc import KMultiAcc
from MCBoost import MCBoost
import pandas as pd
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from helper_functions import MSCE as MSCE
import LSBoost
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# Mega Function that takes in dataset
def achieving_calibration_with_witness(features, labels, baseline_model="Logistic_Regression", witness_metric = 'rbf'):
  scaler = StandardScaler()
  #fit transform data
  features = scaler.fit_transform(features, labels)
  labels = labels.astype('int')

  # initialize lists
  MSCE_all = []
  KME_all = []
  AUC_all = []
  Classification_Errors_all = []
  MSE_all = []

  for seed in range(2, 25, 5):
    X_train_wit_val, X_test, y_train_wit_val, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=seed)

    X_train, X_wit_val, y_train, y_wit_val = train_test_split(
        X_train_wit_val, y_train_wit_val, test_size=0.4, random_state=seed)

    X_wit, X_val, y_wit, y_val = train_test_split(X_wit_val, y_wit_val, test_size=0.4, random_state=seed)

    if baseline_model == "Logistic_Regression":
      model = LogisticRegression(max_iter=10000)
      weak_learner = LogisticRegression(max_iter=10000)
    elif baseline_model == "Decision_Tree":
      model = DecisionTreeClassifier(max_depth = 1)
      weak_learner = DecisionTreeClassifier(max_depth = 1)
    elif baseline_model == "Random_Forest":
      model = RandomForestClassifier(max_depth=2, random_state=seed)
      weak_learner = RandomForestClassifier(max_depth=2, random_state=seed)
    elif baseline_model == "Kernel_SVM":
      model = SVC(gamma="auto", probability = True)
      weak_learner = SVC(gamma="auto", probability = True)
    elif baseline_model == "Naive_Bayes":
      model = GaussianNB()
      weak_learner = GaussianNB()
    elif baseline_model == "NN":
      model = MLPClassifier(random_state=seed, max_iter=30)
      weak_learner = MLPClassifier(random_state=1, max_iter=30)
    else:
      return "Baseline not supported"

    # fit baseline model
    model.fit(X_train, y_train)

    # fit KMAcc 
    kma = KMultiAcc(baseline_model = baseline_model)
    kma.fit_model(X_train, y_train)
    kma.fit(X_wit_val, y_wit_val)

    # Sanity Check delete later!!
    kma.y_test = y_val

    # kMAcc + isotonic calibration
    kMAcc_iso = CalibratedClassifierCV(kma, cv="prefit", method="isotonic")
    kMAcc_iso.fit(X_val, y_val)

    # baseline + isotonic calibration
    base_iso = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    base_iso.fit(X_val, y_val)

    #LSBoost
    LSBoostReg = LSBoost.LSBoostingRegressor(
                                  T = 100,
                                  num_bins = 100,
                                  min_group_size = 5,
                                  global_gamma = .005,
                                  weak_learner=weak_learner,
                                  bin_type = 'distribution',
                                  learning_rate = .1,
                                  initial_model = None,
                                  final_round = True,
                                  center_mean=False)
    LSBoostReg.fit(X_train_wit_val, y_train_wit_val)

    # MCBoost
    f_MCBoost = MCBoost(partition = True, multiplicative = True, init_predictor = gen_preds(model), max_iter = 10)
    f_MCBoost.fit(X_train_wit_val, y_train_wit_val)

    # MCBoost + isotonic calibration
    mcboost_iso = CalibratedClassifierCV(f_MCBoost, cv="prefit", method="isotonic")
    mcboost_iso.fit(X_val, y_val)

    KMEs_itr, MSCEs_itr, AUCs_itr, Classification_Errors_itr, MSEs_itr = mega_compute_corrected(kma.gopt, witness_metric, X_test, y_test, model, kma, \
            LSBoostReg, f_MCBoost, base_iso, kMAcc_iso, mcboost_iso)
    
    MSCE_all.append(MSCEs_itr)
    KME_all.append(KMEs_itr)
    AUC_all.append(AUCs_itr)
    Classification_Errors_all.append(Classification_Errors_itr)
    MSE_all.append(MSEs_itr)
    print("\n")

  return MSCE_all, KME_all, AUC_all, Classification_Errors_all, MSE_all


def run_for_task(features, labels, base_classifiers, prefix):
  data_class = []
  for classifier in base_classifiers:
    MSCE_all, KME_all, AUC_all, Classification_Errors_all, MSE_all = achieving_calibration_with_witness(features, labels, classifier)

    data_class.append((MSCE_all, KME_all, AUC_all, Classification_Errors_all, MSE_all))

  # save data_class
  save_df = pd.DataFrame(data_class)
  filename = prefix + 'all.csv'
  save_df.to_csv(filename)

# def achieving_calibration_with_witness(features, label, baseline_model="Logistic_Regression", witness_metric = 'rbf'):
#   scaler = StandardScaler()
#   #fit transform data
#   features = scaler.fit_transform(features, label)
#   label = label.astype('int')

#   # initialize lists
#   base_msce = []
#   base_KME = []
#   kMAcc_msce = []
#   kMAcc_KME = []
#   LSBoost_msce = []
#   LSBoost_KME = []
#   MCBoost_msce = []
#   MCBoost_KME = []
#   base_iso_msce = []
#   base_iso_KME = []
#   kMacc_iso_msce = []
#   kMAcc_iso_KME = []
#   MCBoost_iso_msce = []
#   MCBoost_iso_KME = []
#   AUC = []

#   for seed in range(2, 25, 5):
#     X_train_wit_val, X_test, y_train_wit_val, y_test = train_test_split(
#         features, label, test_size=0.3, random_state=seed)

#     X_train, X_wit_val, y_train, y_wit_val = train_test_split(
#         X_train_wit_val, y_train_wit_val, test_size=0.4, random_state=seed)

#     X_wit, X_val, y_wit, y_val = train_test_split(X_wit_val, y_wit_val, test_size=0.4, random_state=seed)

#     if baseline_model == "Logistic_Regression":
#       model = LogisticRegression(max_iter=10000)
#       weak_learner = LogisticRegression(max_iter=10000)
#     elif baseline_model == "Decision_Tree":
#       model = DecisionTreeClassifier(max_depth = 1)
#       weak_learner = DecisionTreeClassifier(max_depth = 1)
#     elif baseline_model == "Random_Forest":
#       model = RandomForestClassifier(max_depth=2, random_state=seed)
#       weak_learner = RandomForestClassifier(max_depth=2, random_state=seed)
#     elif baseline_model == "Kernel_SVM":
#       model = SVC(gamma="auto", probability = True)
#       weak_learner = SVC(gamma="auto", probability = True)
#     elif baseline_model == "Naive_Bayes":
#       model = GaussianNB()
#       weak_learner = GaussianNB()
#     elif baseline_model == "NN":
#       model = MLPClassifier(random_state=seed, max_iter=300)
#       weak_learner = MLPClassifier(random_state=1, max_iter=300)
#     else:
#       return "Baseline not supported"

#     # fit baseline model
#     model.fit(X_train, y_train)

#     # fit KMAcc 
#     kma = KMultiAcc(baseline_model = baseline_model)
#     kma.fit_model(X_train, y_train)
#     kma.fit(X_wit_val, y_wit_val, X_wit, y_wit, X_val, y_val)

#     # kMAcc + isotonic calibration
#     kMAcc_iso = CalibratedClassifierCV(kma, cv="prefit", method="isotonic")
#     kMAcc_iso.fit(X_val, y_val)

#     # baseline + isotonic calibration
#     base_iso = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
#     base_iso.fit(X_val, y_val)

#     #LSBoost
#     LSBoostReg = LSBoost.LSBoostingRegressor(
#                                   T = 100,
#                                   num_bins = 100,
#                                   min_group_size = 5,
#                                   global_gamma = .005,
#                                   weak_learner=weak_learner,
#                                   bin_type = 'distribution',
#                                   learning_rate = .1,
#                                   initial_model = None,
#                                   final_round = True,
#                                   center_mean=False)
#     LSBoostReg.fit(X_train_wit_val, y_train_wit_val)

#     # MCBoost
#     f_MCBoost = MCBoost(partition = True, multiplicative = True, init_predictor = gen_preds(model), max_iter = 10)
#     f_MCBoost.fit(X_train_wit_val, y_train_wit_val)

#     # MCBoost + isotonic calibration
#     mcboost_iso = CalibratedClassifierCV(f_MCBoost, cv="prefit", method="isotonic")
#     mcboost_iso.fit(X_val, y_val)

#     base_msce_temp, base_KME_temp, kMAcc_msce_temp, kMAcc_KME_temp, LSBoost_msce_temp, LSBoost_KME_temp, \
#           MCBoost_msce_temp, MCBoost_KME_temp, base_iso_msce_temp, base_iso_KME_temp, kMacc_iso_msce_temp, kMAcc_iso_KME_temp, \
#           MCBoost_iso_msce_temp, MCBoost_iso_KME_temp, AUC_temp = mega_compute_corrected(kma.gopt, witness_metric, X_test, y_test, model, kma, \
#             LSBoostReg, f_MCBoost, base_iso, kMAcc_iso, mcboost_iso)
    
#     base_msce.append(base_msce_temp)
#     base_KME.append(base_KME_temp)
#     kMAcc_msce.append(kMAcc_msce_temp)
#     kMAcc_KME.append(kMAcc_KME_temp)
#     LSBoost_msce.append(LSBoost_msce_temp)
#     LSBoost_KME.append(LSBoost_KME_temp)
#     MCBoost_msce.append(MCBoost_msce_temp)
#     MCBoost_KME.append(MCBoost_KME_temp)
#     base_iso_msce.append(base_iso_msce_temp)
#     base_iso_KME.append(base_iso_KME_temp)
#     kMacc_iso_msce.append(kMacc_iso_msce_temp)
#     kMAcc_iso_KME.append(kMAcc_iso_KME_temp)
#     MCBoost_iso_msce.append(MCBoost_iso_msce_temp)
#     MCBoost_iso_KME.append(MCBoost_iso_KME_temp)
#     AUC.append(AUC_temp)
#     print("\n")

#   return base_msce, base_KME, kMAcc_msce, kMAcc_KME, LSBoost_msce, LSBoost_KME, MCBoost_msce, MCBoost_KME, \
#     base_iso_msce, base_iso_KME, kMacc_iso_msce, kMAcc_iso_KME, MCBoost_iso_msce, MCBoost_iso_KME, AUC
