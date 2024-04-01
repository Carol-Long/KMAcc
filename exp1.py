
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility
from run_KMAcc import run_for_task

base_classifiers = ["Logistic_Regression", "Naive_Bayes", "Random_Forest", "NN", "Decision_Tree"]

# Employment task MA
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["MA"], download=True)
features, labels, group = ACSEmployment.df_to_numpy(acs_data)
prefix = "EMP_MA_"
run_for_task(features, labels, base_classifiers, prefix)

# Income Task IL
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
data = data_source.get_data(states=["IL"], download=True)
features, labels, _ = ACSIncome.df_to_numpy(data)
prefix = "Income_IL_"
run_for_task(features, labels, base_classifiers, prefix)

# Health Public Coverage Task WI
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
data = data_source.get_data(states=["WI"], download=True)
features, labels, _ = ACSPublicCoverage.df_to_numpy(data)
prefix = "Health_WI_"
run_for_task(features, labels, base_classifiers, prefix)

# Employment task Al
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["AL"], download=True)
features, labels, group = ACSEmployment.df_to_numpy(acs_data)
prefix = "EMP_AL_"
run_for_task(features, labels, base_classifiers, prefix)

# Income Task WA
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
data = data_source.get_data(states=["WA"], download=True)
features, labels, _ = ACSIncome.df_to_numpy(data)
prefix = "Income_WA_"
run_for_task(features, labels, base_classifiers, prefix)

# Health Public Coverage Task OH
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
data = data_source.get_data(states=["OH"], download=True)
features, labels, _ = ACSPublicCoverage.df_to_numpy(data)
prefix = "Health_OH_"
run_for_task(features, labels, base_classifiers, prefix)

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
