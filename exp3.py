
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility
from run_KMAcc import run_for_task

base_classifiers = ["Logistic_Regression", "Naive_Bayes", "Random_Forest", "NN"]

# Health Public Coverage Task WI
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
data = data_source.get_data(states=["WI"], download=True)
features, labels, _ = ACSPublicCoverage.df_to_numpy(data)
prefix = "Health_WI_"
run_for_task(features, labels, base_classifiers, prefix)

# Health Public Coverage Task OH
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
data = data_source.get_data(states=["OH"], download=True)
features, labels, _ = ACSPublicCoverage.df_to_numpy(data)
prefix = "Health_OH_"
run_for_task(features, labels, base_classifiers, prefix)
