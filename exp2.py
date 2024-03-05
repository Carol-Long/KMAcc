
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility
from run_KMAcc import run_for_task

base_classifiers = ["Logistic_Regression", "Naive_Bayes", "Random_Forest", "NN"]

# Income Task WA
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
data = data_source.get_data(states=["WA"], download=True)
features, labels, _ = ACSIncome.df_to_numpy(data)
prefix = "Income_WA_"
run_for_task(features, labels, base_classifiers, prefix)

# Income Task IL
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
data = data_source.get_data(states=["IL"], download=True)
features, labels, _ = ACSIncome.df_to_numpy(data)
prefix = "Income_IL_"
run_for_task(features, labels, base_classifiers, prefix)
