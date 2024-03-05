
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility
from run_KMAcc import run_for_task

base_classifiers = ["Logistic_Regression", "Naive_Bayes", "Random_Forest", "NN"]

# ACS Mobility Task NJ
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
data = data_source.get_data(states=["NJ"], download=True)
features, labels, _ = ACSMobility.df_to_numpy(data)
prefix = "Mobility_NJ_"
run_for_task(features, labels, base_classifiers, prefix)

# ACS Mobility Task NY
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
data = data_source.get_data(states=["NY"], download=True)
features, labels, _ = ACSMobility.df_to_numpy(data)
prefix = "Mobility_NY_"
run_for_task(features, labels, base_classifiers, prefix)
