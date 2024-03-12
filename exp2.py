
import pandas as pd
from run_KMAcc import run_for_task

base_classifiers = ["Logistic_Regression", "Naive_Bayes", "Random_Forest", "NN"]
# Enem
file_path = 'data/'
filename = 'enem-50000.pkl'
df = pd.read_pickle(file_path+filename)
df['gradebin'] = df['gradebin'].astype(int)
features, labels= df.drop(columns=['gradebin']).to_numpy(), df['gradebin'].to_numpy()
prefix = "ENEM_"
run_for_task(features, labels, base_classifiers, prefix)
