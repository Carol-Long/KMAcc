
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility
from run_KMAcc import run_for_task

base_classifiers = ["Logistic_Regression", "Naive_Bayes", "Random_Forest", "NN"]

# Employment task MA
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["MA"], download=True)
features, labels, group = ACSEmployment.df_to_numpy(acs_data)
prefix = "EMP_MA_"
run_for_task(features, labels, base_classifiers, prefix)

# Employment task Al
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["AL"], download=True)
features, labels, group = ACSEmployment.df_to_numpy(acs_data)
prefix = "EMP_AL_"
run_for_task(features, labels, base_classifiers, prefix)
