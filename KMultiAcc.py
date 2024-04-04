import numpy as np
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.neural_network import MLPClassifier
import cvxpy as cp
from sklearn.naive_bayes import GaussianNB
from MAccWitness import MAccWitness
from utils import grid_search_params

from utils import compute_calibration_error
from sklearn.model_selection import KFold

class KMultiAcc(BaseEstimator, RegressorMixin):
  def __init__(self, baseline_model="Logistic_Regression"):
    # baseline classifiers
    if baseline_model == "Logistic_Regression":
      self.model = LogisticRegression(max_iter=10000)
    elif baseline_model == "Decision_Tree":
      self.model = DecisionTreeClassifier(max_depth = 1)
    elif baseline_model == "Random_Forest":
      self.model = RandomForestClassifier(max_depth=2, random_state=0)
    elif baseline_model == "Kernel_SVM":
      self.model = SVC(gamma="auto", probability = True)
    elif baseline_model == "Naive_Bayes":
      self.model = GaussianNB()
    elif baseline_model == "NN":
      self.model = MLPClassifier(random_state=1, max_iter=300)
    else:
      return "Baseline not supported"

    self.max_itr_num = 10
    self.lambda_opt_ = []
    self.gamma_opt = []
    self.baseline_model = baseline_model
    self._is_fitted = False
    self._estimator_type = "classifier"
    self.wit_model = []
    self.gopt = 1 # standardize gamma for evaluating final KME
    # for sanity check, delete later
    self.y_test = None
    self.wit = []

  def fit_model(self, X, y):
    self.classes_ = unique_labels(y)
    self.model.fit(X, y)

  def fit(self, X_wit_val, y_wit_val, witness_metric = 'rbf', alpha = .75):
    # find the best parameter (gamma, lambda) for the witness function in each iteration
    X_wit, X_val, y_wit, y_val = train_test_split(X_wit_val, y_wit_val, test_size=0.5, random_state=42)
    lambdas = np.arange(0, 2, 1e-1)

    for i in range(self.max_itr_num):
      if i == 0:
        yhat_proba_wit = self.model.predict_proba(X_wit)[:, 1]
        error_wit = y_wit - yhat_proba_wit
        yhat_proba_val = self.model.predict_proba(X_val)[:, 1]
        error_val = y_val - yhat_proba_val
        yhat_f0 = yhat_proba_val

      # search for optimal rbf kernel param 
      self.gamma_opt.append(grid_search_params(witness_metric, X_val, error_val))

      # define witness on witness set
      wit = MAccWitness(gamma=self.gamma_opt[-1], metric=witness_metric)
      wit_model = make_pipeline(StandardScaler(), wit)
      wit_model.fit(X_wit, error_wit)
      self.wit_model.append(wit_model)
      # for sanity check, delete later
      self.wit.append(wit)
      wit_val = wit_model.predict(X_val)

      # search for best lambda (parameter for the updated predictor)
      calibration_error = np.zeros(len(lambdas)) # KME
      mse_f0 = np.zeros(len(lambdas)) # mean squared error wrt to f_0
      for i in range(len(lambdas)):
        g_val, _ = self.update_proba(yhat_proba_val, lambdas[i], wit_val)
        calibration_error[i] = wit.compute_KME(X_val, y_val, g_val)
        mse_f0[i] = np.mean((g_val-yhat_f0)**2)

      # sort mse_f0, pick the smallest whose calibration_error is less than alpha
      temp_alpha = min(calibration_error)+(np.max(calibration_error)-np.min(calibration_error))/(2+i)
      valid_lambdas = lambdas[calibration_error < temp_alpha]
      min_mse_index = np.argmin(mse_f0[calibration_error < temp_alpha])
      self.lambda_opt_.append(valid_lambdas[min_mse_index])
      print("Optimal lambda:", self.lambda_opt_[-1])

      ## Alternative strategy: QP
      """opt_l, eps = self.solve_qp(yhat_proba_val, y_val, wit_val)
      self.lambda_opt_ = opt_l
      print(min(yhat_proba_val + opt_l * wit_val), max(yhat_proba_val + opt_l * wit_val))
      print(f"Lambda: {opt_l}")
      print(f"{np.sum(eps), max(eps), min(eps)}")"""
    
      # update predicted probabilities using witness
      yhat_proba_wit = self.update_proba(yhat_proba_wit, self.lambda_opt_[-1], wit_model.predict(X_wit))[0]
      error_wit = y_wit - yhat_proba_wit
      yhat_proba_val = self.update_proba(yhat_proba_val, self.lambda_opt_[-1], wit_val)[0]
      error_val = y_val - yhat_proba_val
    self._is_fitted = True
    return self

  def predict_proba(self,X):
    yhat_proba_test = self.model.predict_proba(X)[:,1]
    yhat_f0 = yhat_proba_test
    # subtract witness iteraitvely
    for i in range(self.max_itr_num):
      wit_model = self.wit_model[i]
      wit_test = wit_model.predict(X)
      yhat_proba_test, yhat_test = self.update_proba(yhat_proba_test, self.lambda_opt_[i], wit_test)
      # print("Test Sum of square updates:", np.mean((yhat_proba_test-yhat_f0)**2))
      # sanity check Delete later!
      # print("Test KME: ", self.wit[i].compute_KME(X, self.y_test, yhat_proba_test))
    return np.column_stack((1 - yhat_proba_test, yhat_proba_test))

  def predict(self, X):
    yhat_proba_test = self.model.predict_proba(X)[:,1]
    for i in range(self.max_itr_num):
      wit_model = self.wit_model[i]
      wit_test = wit_model.predict(X)
      yhat_proba_test, yhat_test = self.update_proba(yhat_proba_test, self.lambda_opt_[i], wit_test)
    return yhat_test

  # updated model algorithm using witness
  def update_proba(self, yhat_proba, lambda_, wit_value):
    yhat_proba = yhat_proba + lambda_ * wit_value
    # set all <0 values to 0, all >1 values to 1
    yhat_proba[yhat_proba < 0] = 0
    yhat_proba[yhat_proba > 1] = 1
    yhat_pred = (yhat_proba > 0.5).astype('int')
    return yhat_proba, yhat_pred

  def solve_qp(self, f, y, c, alpha = .02):
    #f is predictor on validation points
    #y is true label validation points
    #c is witness function applied to validation points

    with open('qp.npy', 'wb') as file:
      np.save(file, f)
      np.save(file, y)
      np.save(file, c)

    n = len(f) #n is dim

    #alpha is multiaccuracy constraint

    f.reshape((len(f), 1))

    A = np.row_stack((c.T / n, -1 * c.T / n, np.diag(np.ones(n)), np.diag(np.ones(n))))
    print(f"A: {A.shape}")
    b = np.row_stack((alpha + c.T @ y / n, alpha - c.T @ y / n, np.ones((n, 1)), np.ones((n, 1))))
    print(f"b: {b.shape}")
    print(f"n: {n}")

    print(f"A@f: {(A @ f / n).shape}")
    Bmat = 1 / 2 * A @ A.T
    d = b - (A @ f).reshape((len(b), 1))

    print(f"Bmat: {Bmat.shape}")
    #print(f"Rank of Bmat: {np.linalg.matrix_rank(Bmat)}")
    """u, s, v = np.linalg.svd(A)
    print(f"SVD: {s}")
    plt.hist(s)
    plt.show()
    print("f")
    print(f"Condition number of Bmat: {np.linalg.cond(Bmat)}")
    eigdecomp = np.linalg.eig(Bmat)
    print(f"Notable eig of Bmat: {eigdecomp[0]}")
    print(f"Norm of C: {np.linalg.norm(c)}")
    plt.hist(c)
    plt.show()"""
    print(f"d: {d.shape}")
    print(f"f: {f.shape}")

    L = cp.Variable((2 * n + 2, 1))
    print(f"L: {L.shape}")
    constraints = [0 <= L]
    objective = cp.Minimize(cp.quad_form(L, cp.Parameter(shape=Bmat.shape, value = Bmat, PSD=True)) + d.T @ L)
    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver=cp.SCS)
    val = L.value
    l = (val[1] - val[0]) / n
    print(f"Lambda dual: {val[0], val[1]}")
    eps = (val[n+2:] - val[2:n+2])
    return l, eps
