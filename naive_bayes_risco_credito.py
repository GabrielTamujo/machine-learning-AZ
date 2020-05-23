# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:00:13 2019

@author: gabri
"""

import pandas as pd
base = pd.read_csv('risco-credito.csv')

#Divisão do dataset
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

#Pré-processamento
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
previsores[:, 0] = labelEncoder.fit_transform(previsores[:, 0])
previsores[:, 1] = labelEncoder.fit_transform(previsores[:, 1])
previsores[:, 2] = labelEncoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelEncoder.fit_transform(previsores[:, 3])

#Treinamento
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)

#Fazendo previsões
resultado = classificador.predict([[0, 0, 1, 2], [3, 0, 0, 0]])