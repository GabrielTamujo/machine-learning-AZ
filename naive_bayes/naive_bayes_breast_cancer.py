# -*- coding: utf-8 -*-
import pandas as pd

base = pd.read_csv("../datasets/breast-cancer-wisconsin.csv")

base.drop(index=base.loc[base['1.3'] == '?'].index, inplace=True)

previsores = base.iloc[:, 1:9].values
classe = base.iloc[:, 10].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

#Aplicação do Naive Bayes
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

#Análise de Precisão
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)