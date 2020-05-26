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
- Contínua: engloba o conjunto de números reais que podem representar valores em dinheiro, temperatura, altura, peso ou qualquer dimensão.
- Discreta: engloba, geralmente, o conjunto de números finitos inteiros que representam uma categoria ou um grupo.

## Variáveis Categóricas
As variváveis categóricas englobam as variváveis do tipo String. Elas se duvidem em dois sub-grupos:
- Nominal: engloba categorias não mensuráveis, ou seja, que não possui uma lógica ordinal. Ex.: cores, gênero, etc.
- Ordinal: engloba categorias que possuem uma lógica de ordenação. Ex.: Tamanho P, M e G. 

# Base de Dados de Crédito | credit-data.csv
Um caso bastante frequente onde aplica-se algoritmos de Machine Learning consiste em análise bancária de crédito. Ou seja, a partir de uma série de parâmetros, tomar a decisão de se o empréstimo será liberado ou não para determinada pessoa. Este é o primeiro caso que será trabalhado.

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

Para este tipo de problema, geralmente, recomenda-se trabalhar com o Escalonamento por Padronização (Standardisation), que define os valores através da fórmula abaixo:

![Standardisation](https://user-images.githubusercontent.com/30511610/82903664-a5afc900-9f37-11ea-86be-8b5527529a5b.png)

Execute a célula abaixo para aplicar o escalonamento nos previsores:

```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
```

## Divisão da Base de Dados em Treino e Teste
Ao finalizar o pré-processamento de dados, é necessário separar os registros entre treinamento e teste para a aplicação dos algoritmos.

Execute:

```
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)
```

Com isso, finalizamos o pré-processamento da base de dados de crédito. 

# Base de Dados do Censo | census.csv
A base de dados do [Censo](https://archive.ics.uci.edu/ml/datasets/Adult) é uma base um pouco mais robusta em quantidade de dados que visa prever se um indivíduo recebe >50K ou <=50K por ano. 

```
age,workclass,final-weight,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loos,hour-per-week,native-country,income
39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K
```
Esta base possui 15 colunas descritas abaixo:

- age: variavél numérica contínua.
- workclass: variável categórica nominal, podendo ser do tipo Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: variavél numérica contínua.
- education: variável categórica nominal, podendo ser do tipo Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- education-num: variavél numérica discreta.
- marital-status: variável categórica nominal, podendo ser do tipo Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: variável categórica nominal, podendo ser do tipo Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: variável categórica nominal, podendo ser do tipo Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: variável categórica nominal, podendo ser do tipo White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: variável categórica nominal, podendo ser do tipo Female, Male.
- capital-gain: variavél numérica discreta.
- capital-loss: variavél numérica discreta.
- hours-per-week: variavél numérica discreta.
- native-country: variável categórica nominal, podendo ser do tipo United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
- income: varivável categórica ordinal

Ao verificar as condições dos dados, utilizando abordagens simples como `base.describe()` e `ps.isnull()`, perceberemos que este dataset não necessita correções quanto a valores faltantes ou inconsistentes. 

## Transformação de Varíaveis Categóricas em Variáveis Numéricas

Para se trabalhar com Machine Learning em bases como essa, é necessário criar referências numéricas para as variáveis categóricas, uma vez que textos não significam nada para um computador. Porém, é importante nesta etapa possuir pleno domínio sob a diferenciação entre variáveis categóricas nominais e ordinais para definir esta transformação. Se nós simplesmente definirmos uma sequência de números para cada variável nominal, ou seja, uma variável que não possui semântica atrelada a uma ordem entre categorias, o algoritmo pode acabar considerando errôneamente as sequências numéricas mais altas como as mais importantes. Dessa forma, é preciso criar variáveis do tipo Dummy. Isto é, cada categoria de uma determinada verívavel nominal deverá se tornar uma nova coluna da tabela, atribuindo valores de 0 ou 1 para ativar ou não esta categoria em um determinado registro. 

![dummy](https://user-images.githubusercontent.com/30511610/82900220-924e2f00-9f32-11ea-997a-edd7f633c6a8.png)

O primeiro passo para tal consiste em, novamente, dividr o dataset em previsores e classe:

```
import pandas as pd
base = pd.read_csv('census.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
```

As classes que o [Scikit-Learn](https://scikit-learn.org/stable/) oferece para trabalhar com este tipo de conversão são a `LabelEncoder` e `OneHotEncoder`.

Execute a célula abaixo para a conversão:

```
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],remainder='passthrough')
previsores = column_tranformer.fit_transform(previsores).toarray()
```

A coluna classe, uma vez que possui apenas os atributos >50K ou <=50K, deverá tornar-se referências númericas 0 e 1. Para tal, deverá ser utilizado a ferramenta `LabelEncoder()` que, ao passar como atributo o DataFrame classe, codificará as categorias da maneira esperada.

```
labelencorder_classe = LabelEncoder()
classe = labelencorder_classe.fit_transform(classe)
```

## Escalonamento dos valores
Não diferente da primeira base dados, aqui também se faz necessário o escalonamento dos atributos. Um dúvida frequente é se deve ou não aplicar o escalonamento nas variáveis do tipo Dummy, uma vez que elas apenas indicam a presença do atributo ou não e, se aplicarmos o escalonamento, poderá dificultar a nossa interpretação dos dados. Porém, o que para nós parece não fazer sentido ao visualizar, para os algoritmos costuma ser uma diferença essencial, visto que tende a afetar bastante o resultado final de acertividade. 

Para aplicar o escalonamento em todos os atributos execute:

```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
```

## Divisão da Base de Dados em Treino e Teste
Ao finalizar o pré-processamento de dados, é necessário separar os registros entre treinamento e teste para a aplicação dos algoritmos.

Execute:

```
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)
```

Com isso, finalizamos o pré-processamento da base de dados do Censo.


