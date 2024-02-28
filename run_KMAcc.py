import sys
sys.path.append('Level-Set-Boosting')

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSTravelTime, ACSPublicCoverage, ACSMobility
from folktables import BasicProblem
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cluster import KMeans
from sklearn.calibration import CalibratedClassifierCV
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper_functions import MSCE as MSCE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
import LSBoost
import helper_functions as hf
from itertools import product
from sklearn.neural_network import MLPClassifier

import seaborn as sns
import cvxpy as cp
from sklearn.naive_bayes import GaussianNB
