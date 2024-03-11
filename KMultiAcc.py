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

    self.wit_model = None
    self.opt_lambda_ = None
    self.baseline_model = baseline_model
    self._is_fitted = False
    self.wit_model_ = None
    self.gopt = None
    self._estimator_type = "classifier"
    
  def fit_model(self, X, y):
    self.model.fit(X, y)

  def fit(self, X_train_wit_val, y_wit_train_val, X_wit = None, y_wit = None, X_val = None, y_val = None, witness_metric = 'rbf', alpha = .01):
    if X_val is None and X_wit is None:
      X_train, X_wit_val, y_train, y_wit_val = train_test_split(X_train_wit_val, y_wit_train_val, test_size=0.5) # Use test_size = .5 for non-qp
      X_wit, X_val, y_wit, y_val = train_test_split(X_wit_val, y_wit_val, test_size=0.5) # Use test_size = .5 for non-qp
      self.fit_model(X_train, y_train)

    self.classes_ = unique_labels(y_wit)

    yhat_proba_wit = self.model.predict_proba(X_wit)[:, 1]
    error_wit = y_wit - yhat_proba_wit

    yhat_proba_val = self.model.predict_proba(X_val)[:, 1]
    error_val = y_val - yhat_proba_val

    # search for optimal rbf kernel param using witness set
    self.gopt = grid_search_params(witness_metric, X_wit, error_wit)

    # define witness on witness set
    wit = MAccWitness(gamma=self.gopt, metric=witness_metric)

    self.wit_model = make_pipeline(StandardScaler(), wit)
    self.wit_model.fit(X_wit, error_wit)

    # search for best lambda using validation set (parameter for the updated predictor)
    wit_val = self.wit_model.predict(X_val)

    ## Put in QP
    """opt_l, eps = self.solve_qp(yhat_proba_val, y_val, wit_val)
    self.lambda_opt = opt_l
    print(min(yhat_proba_val + opt_l * wit_val), max(yhat_proba_val + opt_l * wit_val))
    print(f"Lambda: {opt_l}")
    print(f"{np.sum(eps), max(eps), min(eps)}")"""
    # lambdas = np.arange(0, .1, 0.003) # ACS datasets
    lambdas = np.arange(0, 1, 1e-3)
    divergence = np.zeros(len(lambdas))
    calibration_error = np.zeros(len(lambdas))

    self.lambda_opt = None
    for i in range(len(lambdas)):
      g_val, g_val_pred = self.update_proba(yhat_proba_val, lambdas[i], wit_val)
      divergence[i] = np.linalg.norm(yhat_proba_val - g_val)
      calibration_error[i] = compute_calibration_error(wit_val, y_val, g_val)
    '''
    for i in np.argsort(divergence):
      if calibration_error[i] < alpha:
        self.lambda_opt = lambdas[i]
        break
    '''
    if self.lambda_opt is None: self.lambda_opt = lambdas[np.nanargmin(calibration_error)]
    print("Optimal lambda:", self.lambda_opt)


    self._is_fitted = True
    self.wit_model_ = self.wit_model

    return self

  def predict_proba(self,X):
    if self.wit_model == None:
      raise ValueError("No Witness was fit!")

    yhat_proba_test = self.model.predict_proba(X)[:,1]
    wit_test = self.wit_model.predict(X)
    yhat_proba_updated, yhat_updated = self.update_proba(yhat_proba_test, self.lambda_opt, wit_test)
    return np.column_stack((1 - yhat_proba_updated, yhat_proba_updated))

  def predict(self, X):
    if self.wit_model == None:
      raise ValueError("No Witness was fit!")
    yhat_proba_test = self.model.predict_proba(X)[:,1]
    wit_test = self.wit_model.predict(X)
    yhat_proba_updated, yhat_updated = self.update_proba(yhat_proba_test, self.lambda_opt, wit_test)
    return yhat_updated

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
