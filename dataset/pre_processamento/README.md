# Pré-processamento de Dados
Uma etapa essencial na aplicação de algoritmos de Machine Learning consiste no pré-processamento dos dados. Isto é, a partir de uma base de dados bruta e, pela aplicação de uma série de metodologias, torná-la propícia a aplicação das técnicas de Machine Learning. Muitas são as metodologias possíveis e, para tal, as bibliotecas [Pandas](https://pandas.pydata.org/) e [Scikit-Learn](https://scikit-learn.org/stable/) oferecem uma série de ferramentas que tornam prático este processo. 
Alguns objetivos frequentes desta etapa:
- Corrigir valores faltantes
- Corrigir valores inconsistentes 
- Escalonamento de atributos numéricos
- Transformação de variáveis categóricas (Variáveis do tipo String p/ referencia numérica)
- Divisão do Dataset em Base de Treinamento e Base de Teste

# Tipos de Variáveis
Dentre as variáveis consideradas nos algoritmos de Machine Learning, existem dois grupos com sub-grupos anexados. A diferenciação e o conhecimento destas nomenclaturas é essencial para a aplicação das técnicas de aprendizagem de máquina.

## Variáveis Numéricas
As variáveis numéricas, logicamente, englobam os tipos de variáveis com valor numérico. Esse tipo se divide em dois sub-grupos:
- Contínua: Engloba o conjunto de números reais que podem representar valores em dinheiro, temperatura, altura, peso ou qualquer dimensão.
- Discreta: Engloba, geralmente, o conjunto de números finitos inteiros que representam uma categoria ou um grupo.

## Variáveis Categóricas
As variváveis categóricas englobam as variváveis do tipo String. Elas se duvidem em dois sub-grupos:
- Nominal: Engloba categorias não mensuráveis, ou seja, que não possui uma lógica ordinal. Ex.: cores, gênero, etc.
- Ordinal: Engloba categorias que possuem uma lógica de ordenação. Ex.: Tamanho P, M e G. 

# Base de Dados de Crédito | credit-data.csv
Um caso bastante frequente onde aplica-se algoritmos de Machine Learning consiste em análise bancária de crédito. Ou seja, a partir de uma série de parâmetros, tomar a decisão de se o empréstimo será liberado ou não para determinada pessoa. Este é o primeiro case que será trabalhado.

``` 
clientid,income,age,loan,default
1,66155.9250950813,59.017015066929204,8106.53213128514,0
2,34415.1539658196,48.11715310486029,6564.745017677379,0
3,57317.1700630337,63.10804949188599,8020.953296386469,0
...
```
Esta base possui 5 colunas:

- clienteid: O id do cliente, uma variável categórica nominal;
- income: a renda anual do cliente, uma variável numérica contínua;
- age: idade do cliente, uma varíavel numérica contínua;
- loan: o quanto ele solicitou de empréstimo, uma variável numérica contínua;
- default: um caso de variável numérica discreta, onde 0 representa os clientes que não efeturam o pagamento e 1 os que pagaram.

Para carregar o DataFrame, execute:
```
import pandas as pd
base = pd.read_csv('credit-data.csv') 
```

Para uma rápida visualização de algumas estatísticas da base de dados, execute:

```
base.describe()
```

## Tratando Valores Inconsistentes
Ao executar o comando `base.describe()`, é possível visualizar algumas inconsistências nos dados, como o registro de idade mínima ser igual a `-52`. Este tipo de inconsistência costuma gerar problemas quando se fala em aprendizagem de máquina, visto que é possível que os algoritmos encontrem correlações entre estes registro que irão afetar as previsões e a acertividade.

Para buscar todos os registros que possuem idade negativa, execute o seguinte comando:
```
base.loc[base['age'] < 0]
```

Neste tipo de situação existe uma série de decisões possíveis. Vejamos algumas delas.

Para apagar os registros com problemas, execute:
```
base.drop(base[base.age < 0].index, inplace=True)
```

Para preencher os valores inconsistentes com a média de idade, execute:
```
mean_age = base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = mean_age
```

## Tratando Valores Nulos
Outra possibilidade bastante frequente consiste em registros com atributos nulos. Esta inconsistência costuma causar uma série de problemas se não tratada.

Para uma rápida busca por registros com valores nulos, execute:

```
base.loc[pd.isnull(base['age'])] 
```

Para tratar estes valores a biblioteca [Scikit-Learn](https://scikit-learn.org/stable/) oferece ferramentas. Mas, para tal, é necessário primeiramente separar os atributos previsores e classe (default) do dataset:

```
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values
```
Nota-se que a coluna id do cliente fora desconsiderada dentre os atributos previsores, visto que o mesmo não tem relevância na tomada de decisão sobre a liberação de crédito.

O tratamento destes valores ocorre através do pacote `SimpleImputer`. Execute:

```
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])
```

## Escalonamento dos valores
Ao visualizarmos e comparamos os valores dos atributos que representam a renda e a idade, notaremos uma grande diferença de escala: os valores de renda são muito maiores que os valores para a idade. Isto tende a se tornar um problema, caso não tratado, principalmente para algoritmos baseados em [Distância Euclidiana](https://pt.wikipedia.org/wiki/Dist%C3%A2ncia_euclidiana), tal como o [KNN](https://medium.com/brasil-ai/knn-k-nearest-neighbors-1-e140c82e9c4e), uma vez que os atributos de maior valor podem ser considerados erroneamente mais importantes.

Para este tipo de problema, geralmente, recomenda-se trabalhar com o Escalonamento por Padronização (Standardisation), que define os valores através da formula abaixo:

![Standardisation](https://user-images.githubusercontent.com/30511610/82810590-39fd2b80-9e65-11ea-9910-cfbe489123e2.png)

Execute a célula abaixo para aplicar o escalonamento nos previsores:

```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
```