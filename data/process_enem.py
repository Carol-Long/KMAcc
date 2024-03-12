#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:05:39 2024

@author: carollong
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def load_enem20(file_path, filename, features, grade_attribute, n_sample, n_classes, multigroup=False):
    ## load csv
    df = pd.read_csv(file_path+filename, encoding='cp860', sep=';')
    print('Original Dataset Shape:', df.shape)

    ## Remove all entries that were absent or were eliminated in at least one exam
    ix = ~df[['TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT']].applymap(lambda x: False if x == 1.0 else True).any(axis=1)
    df = df.loc[ix, :]

    ## Remove "treineiros" -- these are individuals that marked that they are taking the exam "only to test their knowledge". It is not uncommon for students to take the ENEM in the middle of high school as a dry run
    df = df.loc[df['IN_TREINEIRO'] == 0, :]

    ## drop eliminated features
    df.drop(['TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT', 'IN_TREINEIRO'], axis=1, inplace=True)

    ## subsitute race by names
    # race_names = ['N/A', 'Branca', 'Preta', 'Parda', 'Amarela', 'Indigena']
    race_names = [np.nan, 'Branca', 'Preta', 'Parda', 'Amarela', 'Indigena']
    df['TP_COR_RACA'] = df.loc[:, ['TP_COR_RACA']].applymap(lambda x: race_names[x]).copy()

    ## remove repeated exam takers
    ## This pre-processing step significantly reduces the dataset.
    # df = df.loc[df.TP_ST_CONCLUSAO.isin([1,2])]
    df = df.loc[df.TP_ST_CONCLUSAO.isin([1])] 

    ## select features
    df = df[features]

    ## Dropping all rows or columns with missing values
    df = df.dropna()
    ## Creating racebin & gradebin & sexbin variable
    df['gradebin'] = construct_grade(df, grade_attribute, n_classes)
    if multigroup:
        df['racebin'] = construct_race(df, 'TP_COR_RACA')
    else:
        df['racebin'] =np.logical_or((df['TP_COR_RACA'] == 'Branca').values, (df['TP_COR_RACA'] == 'Amarela').values).astype(int)
    df['sexbin'] = (df['TP_SEXO'] == 'M').astype(int)

    df.drop([grade_attribute[0], 'TP_COR_RACA', 'TP_SEXO'], axis=1, inplace=True)

    ## except for Q005, replace alphabetical values with numerical values
    ## Q005 is 'Including yourself, how many people currently live in your household?'
    question_vars = ['Q00'+str(x) for x in range(1,10)]
    for q in question_vars:
        if q != 'Q005':
            if df[q].dtype == object:
                letter_dict = {chr(65+i): i+1 for i in range(26)}
                df[q] = df[q].replace(letter_dict)
            
    ## check if age range ('TP_FAIXA_ETARIA') is within attributes
    if 'TP_FAIXA_ETARIA' in features:
        q = 'TP_FAIXA_ETARIA'
        if df[q].dtype == object:
            letter_dict = {chr(65+i): i+1 for i in range(26)}
            df[q] = df[q].replace(letter_dict)

    ## encode SG_UF_PROVA (state where exam was taken) as numbers
    label_encoder = LabelEncoder()
    df['SG_UF_PROVA'] = label_encoder.fit_transform(df['SG_UF_PROVA'])
    df = df.dropna()

    ## Scaling ##
    scaler = MinMaxScaler()
    scale_columns = list(set(df.columns.values) - set(['gradebin', 'racebin']))
    df[scale_columns] = pd.DataFrame(scaler.fit_transform(df[scale_columns]), columns=scale_columns, index=df.index)
    # print('Preprocessed Dataset Shape:', df.shape)

    if n_sample != "full":
        df = df.sample(n=min(n_sample, df.shape[0]), axis=0, replace=False)
    df['gradebin'] = df['gradebin'].astype(int)
    print('Transformed Dataset Shape:', df.shape)
    return df

def construct_grade(df, grade_attribute, n):
    v = df[grade_attribute[0]].values
    quantiles = np.nanquantile(v, np.linspace(0.0, 1.0, n+1))
    return pd.cut(v, quantiles, labels=np.arange(n))

def construct_race(df, protected_attribute):
    race_dict = {'Branca': 1, 'Preta': 2, 'Parda': 3, 'Amarela': 4, 'Indigena': 5} # changed to match ENEM 2020 numbering
    return df[protected_attribute].map(race_dict)

## ENEM-2022
enem_path = '' #changed to 2020
enem_file = 'MICRODADOS_ENEM_2022.csv' 
label = ['NU_NOTA_CH'] ## Labels could be: NU_NOTA_CH=human science, NU_NOTA_LC=languages&codes, NU_NOTA_MT=math, NU_NOTA_CN=natural science
group_attribute = ['TP_COR_RACA','TP_SEXO']
# question_vars = ['Q00'+str(x) if x<10 else 'Q0' + str(x) for x in range(1,25)]
question_vars = ['Q00'+str(x) for x in range(1,10)]
domestic_vars = ['SG_UF_PROVA', 'TP_FAIXA_ETARIA'] #changed for 2020
all_vars = label+group_attribute+question_vars+domestic_vars
n_classes = 2
enem_size = "full" #or 50000
fname = 'enem-' + str(enem_size) + '.pkl'
df = load_enem20(enem_path, enem_file, all_vars, label, enem_size, n_classes, multigroup=False)
df.to_pickle(fname)

















