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
def achieving_calibration_with_witness(features, label, baseline_model="Logistic_Regression", witness_metric = 'rbf'):
  scaler = StandardScaler()
  #fit transform data
  features = scaler.fit_transform(features, label)
  label = label.astype('int')

  base_msce = []
  base_KCE = []
  kmulcal_msce = []
  kmulcal_KCE = []
  lsboost_msce = []
  lsboost_KCE = []
  base_msce_binned = []
  base_binned_KCE = []
  kmulcal_msce_binned = []
  kmulcal_binned_KCE = []
  MCBoost_msce = []
  MCBoost_KCE = []
  isotonic_msce = []
  isotonic_KCE = []
  sigmoid_msce = []
  sigmoid_KCE = []
  ablated_msce = []
  ablated_KCE = []
  AUC = []
  MCBoost_iso_msce = []
  MCBoost_iso_KCE = []

  save_np = []
  for seed in range(2, 50, 5):

    X_train_wit_val, X_test, y_train_wit_val, y_test = train_test_split(
        features, label, test_size=0.3, random_state=seed)

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
      model = MLPClassifier(random_state=seed, max_iter=300)
      weak_learner = MLPClassifier(random_state=1, max_iter=300)
    else:
      return "Baseline not supported"


    kma = KMultiAcc(baseline_model = baseline_model)
    kma.fit_model(X_train, y_train)
    kma.fit(X_wit_val, y_wit_val, X_wit, y_wit, X_val, y_val)

    yhat_proba_test = kma.model.predict_proba(X_test)[:,1]
    yhat_proba_wit = kma.model.predict_proba(X_wit)[:,1]
    wit_test = kma.wit_model.predict(X_test)
    wit_witset = kma.wit_model.predict(X_wit)
    g_test, g_test_pred = update_proba(yhat_proba_test, kma.lambda_opt, wit_test)
    g_wit, g_wit_pred = update_proba(yhat_proba_wit, kma.lambda_opt, wit_witset)
    # validation dt
    wit_val = kma.wit_model.predict(X_val)
    yhat_proba_val = kma.model.predict_proba(X_val)[:,1]
    g_val, g_val_pred = update_proba(yhat_proba_val, kma.lambda_opt, wit_val)

    # Ours w isotonic calibration
    model_isotonic = CalibratedClassifierCV(kma, cv="prefit", method="isotonic")
    model_isotonic.fit(X_val, y_val)
    prob_pos_isotonic = model_isotonic.predict_proba(X_test)[:, 1]
    wit_prob_pos_isotonic = model_isotonic.predict_proba(X_wit)[:, 1]

    # Ours w sigmoid calibration
    model_sigmoid = CalibratedClassifierCV(kma, cv="prefit", method="sigmoid")
    model_sigmoid.fit(X_val, y_val)
    prob_pos_sigmoid = model_sigmoid.predict_proba(X_test)[:, 1]

    #Ablated Isotonic Calibration
    model.fit(X_train, y_train)
    ablated_isotonic = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    ablated_isotonic.fit(X_val, y_val)
    prob_pos_ablated = ablated_isotonic.predict_proba(X_test)[:, 1]
    wit_prob_pos_ablated = ablated_isotonic.predict_proba(X_wit)[:, 1]


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

    training_predictions = LSBoostReg.predict(X_train)
    test_predictions = LSBoostReg.predict(X_test)
    wit_predictions_LS = LSBoostReg.predict(X_wit)
    # MCBoost
    model.fit(X_train, y_train)
    mcrf1 = MCBoost(partition = True, multiplicative = True, init_predictor = gen_preds(model), max_iter = 10)
    mcrf1.fit(X_train_wit_val, y_train_wit_val)

    MCBoost_test = mcrf1.predict_proba(X_test)[:, 1]
    MCBoost_wit = mcrf1.predict_proba(X_wit)[:, 1]

    # MCBoost w isotonic calibration
    mcboost_isotonic = CalibratedClassifierCV(mcrf1, cv="prefit", method="isotonic")
    mcboost_isotonic.fit(X_val, y_val)
    prob_test_mcboost_isotonic = mcboost_isotonic.predict_proba(X_test)[:, 1]
    prob_wit_mcboost_isotonic = mcboost_isotonic.predict_proba(X_wit)[:, 1]    

    base_msce_temp, base_KCE_temp, kmulcal_msce_temp, kmulcal_KCE_temp, lsboost_msce_temp, lsboost_KCE_temp, \
         base_msce_binned_temp, base_binned_KCE_temp, kmulcal_msce_binned_temp, kmulcal_binned_KCE_temp, \
         MCBoost_msce_temp, MCBoost_KCE_temp, isotonic_cal_msce_temp, isotonic_cal_KCE_temp, ablated_msce_temp, ablated_KCE_temp, AUC_temp, \
         MCBoost_iso_msce_temp, MCBoost_iso_KCE_temp \
    = mega_compute_corrected(kma.gopt, witness_metric, wit_test, X_wit, y_wit, X_test, y_test, wit_predictions_LS, test_predictions, yhat_proba_test,  \
                                  g_wit, g_test, MCBoost_test, MCBoost_wit, baseline_model, wit_prob_pos_isotonic, prob_pos_isotonic, \
                                  wit_prob_pos_ablated, prob_pos_ablated, prob_test_mcboost_isotonic, prob_wit_mcboost_isotonic)

    # base_msce_temp, base_KCE_temp, kmulcal_msce_temp, kmulcal_KCE_temp, lsboost_msce_temp, \
    # lsboost_KCE_temp, base_msce_binned_temp, base_binned_KCE_temp, kmulcal_msce_binned_temp, kmulcal_binned_KCE_temp, \
    # MCBoost_msce_temp, MCBoost_KCE_temp, isotonic_cal_msce_temp, isotonic_cal_KCE_temp, \
    # sigmoid_cal_msce_temp, sigmoid_cal_kce_temp, ablated_msce_temp, ablated_KCE_temp, AUC_temp \
    #                           = mega_compute(wit_test, y_test, yhat_proba_test, test_predictions, \
    #                                           g_test, MCBoost_test, baseline_model, prob_pos_isotonic, \
    #                                           prob_pos_sigmoid, prob_pos_ablated)

    base_msce.append(base_msce_temp)
    base_KCE.append(base_KCE_temp)
    kmulcal_msce.append(kmulcal_msce_temp)
    kmulcal_KCE.append(kmulcal_KCE_temp)
    lsboost_msce.append(lsboost_msce_temp)
    lsboost_KCE.append(lsboost_KCE_temp)
    base_msce_binned.append(base_msce_binned_temp)
    base_binned_KCE.append(base_binned_KCE_temp)
    kmulcal_msce_binned.append(kmulcal_msce_binned_temp)
    kmulcal_binned_KCE.append(kmulcal_binned_KCE_temp)
    MCBoost_msce.append(MCBoost_msce_temp)
    MCBoost_KCE.append(MCBoost_KCE_temp)
    isotonic_msce.append(isotonic_cal_msce_temp)
    isotonic_KCE.append(isotonic_cal_KCE_temp)
    MCBoost_iso_msce.append(MCBoost_iso_msce_temp)
    MCBoost_iso_KCE.append(MCBoost_iso_KCE_temp)
    # sigmoid_msce.append(sigmoid_cal_msce_temp)
    # sigmoid_KCE.append(sigmoid_cal_kce_temp)
    ablated_msce.append(ablated_msce_temp)
    ablated_KCE.append(ablated_KCE_temp)
    AUC.append(AUC_temp)
    print("\n")

  return base_msce, base_KCE, kmulcal_msce, kmulcal_KCE, lsboost_msce, lsboost_KCE, \
         base_msce_binned, base_binned_KCE, kmulcal_msce_binned, kmulcal_binned_KCE, \
         MCBoost_msce, MCBoost_KCE, isotonic_msce, isotonic_KCE, sigmoid_msce, sigmoid_KCE, \
         ablated_msce, ablated_KCE, AUC, MCBoost_iso_msce, MCBoost_iso_KCE


def run_for_task(features, labels, base_classifiers, prefix):
  data_class = []
  for classifier in base_classifiers:
    base_msce, base_KCE, kmulcal_msce, kmulcal_KCE, lsboost_msce, lsboost_KCE, base_msce_binned, base_binned_KCE, \
    kmulcal_msce_binned, kmulcal_binned_KCE, MCBoost_msce, MCBoost_KCE, isotonic_msce, \
    isotonic_KCE, sigmoid_msce, sigmoid_KCE, ablated_msce, ablated_KCE, AUC, MCBoost_iso_msce, MCBoost_iso_KCE = achieving_calibration_with_witness(features, labels, classifier)

    data_class.append((base_msce, base_KCE, kmulcal_msce, kmulcal_KCE, lsboost_msce, lsboost_KCE, base_msce_binned, base_binned_KCE, \
    kmulcal_msce_binned, kmulcal_binned_KCE, MCBoost_msce, MCBoost_KCE, isotonic_msce, \
    isotonic_KCE, sigmoid_msce, sigmoid_KCE, ablated_msce, ablated_KCE, AUC, MCBoost_iso_msce, MCBoost_iso_KCE))

    # plotter(base_msce, base_KCE, kmulcal_msce, kmulcal_KCE, lsboost_msce, lsboost_KCE, base_msce_binned, base_binned_KCE, \
    # kmulcal_msce_binned, kmulcal_binned_KCE, MCBoost_msce, MCBoost_KCE, isotonic_msce, \
    # isotonic_KCE, sigmoid_msce, sigmoid_KCE, ablated_msce, ablated_KCE, AUC, classifier, prefix, MCBoost_iso_msce, MCBoost_iso_KCE)

  # save data_class
  save_df = pd.DataFrame(data_class)
  filename = prefix + 'all.csv'
  save_df.to_csv(filename)



