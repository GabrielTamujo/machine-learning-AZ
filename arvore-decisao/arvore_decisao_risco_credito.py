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

from sklearn.tree import DecisionTreeClassifier, export
classificador = DecisionTreeClassifier(criterion="entropy")
classificador.fit(previsores, classe)

print(classificador.feature_importances_)

