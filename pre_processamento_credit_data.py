# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#****** IMPORTANDO DADOS
base = pd.read_csv('credit-data.csv')
base.describe()

# ***** TRATAMENTO DE VALORES INVÁLIDOS
base.loc[base['age'] < 0]
# apagar a coluna
base.drop('age', 1, inplace=True)
# apagar somente os registros com problema
base.drop(base[base.age < 0].index, inplace=True)
# preencher os valores manualmente
# preencher os valores com a média
base.mean()
base['age'].mean()
base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = 40.92

#****** TRATAMENTO DE VALORES NULOS
pd.isnull(base['age']) #lista todos os campos e mostra os nulos
base.loc[pd.isnull(base['age'])] #lista apenas os nulos

#separando tabelas
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

#Tratando os nulos
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:,0:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)