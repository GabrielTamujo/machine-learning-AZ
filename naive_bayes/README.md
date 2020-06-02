# Naive Bayes Classifier
Conforme [Steve Liu](https://towardsdatascience.com/@stevhliu) em [The Naive Bayes Classifier](https://towardsdatascience.com/the-naive-bayes-classifier-caaf5b01635e), [Thomas Bayes](https://en.wikipedia.org/wiki/Thomas_Bayes) (1701–1761) formulou seu famoso teorema em resposta a David Hume, que afirmava que evidências inerentemente falíveis são provas insuficientes contra leis naturais, tal como testemunhas oculares não podem provar um milagre. O que realmente interessava a Bayes era responder quanta evidência seria necessária para nos convencer de que algo é uma probabilidade por mais improvável que seja. E, ao fazê-lo, surgiu uma equação que nos permite atualizar nossas crenças com novas evidências. Seu artigo, [An Essay towards solving a Problem in the Doctrine of Chances](https://en.wikipedia.org/wiki/An_Essay_towards_solving_a_Problem_in_the_Doctrine_of_Chances), foi publicado e editado após a sua morte por seu amigo Richard Price, que acreditava que o Teorema de Bayes corroborava a ideia da existência de Deus: 

*"The purpose I mean is, to shew what reason we have for believing that there are in the constitution of things fixt laws according to which things happen, and that, therefore, the frame of the world must be the effect of the wisdom and power of an intelligent case; and thus to confirm the argument taken from final causes for the existence of the Deity."* (Philosophical Transactions of the Royal Society of London, 1763)

Naive Bayes Classifiers, atualmente, consiste em um grupo de potentes classificadores probabilíticos derivados de algoritmos que implementam o Teorema de Bayes.  

Aplicações clássicas:
- Filtros de Spam
- Mineração de emoções
- Separação de documentos
- Sistema de Recomendação
- Multi-class Prediction

Referências complementares:
- [The Optimality of Naive Bayes](http://www.cs.unb.ca/~hzhang/publications/FLAIRS04ZhangH.pdf)
- [Thoughtful Machine Learning | Cap IV](http://feineigle.com/static/books/2018/thoughtful_machine_learning_python/Thoughtful_Machine_Learning_with_Python_-_A_Test-Driven_Approach.pdf)

# Teorema de Bayes
O teorema de Bayes é um corolário da lei da probabilidade total, expresso matematicamente na forma da seguinte equação:
<p align="center">
  <img src="https://user-images.githubusercontent.com/30511610/83259066-c8d3b600-a18d-11ea-9a07-92616f6bf43d.png">
</p>

[Este vídeo](https://www.youtube.com/watch?v=HZGCoVF3YvM) traz uma fantástica explicação da lógica por trás deste teorema.

# O Algoritmo
O processo de treinamento do Naive Bayes Classifier nada mais é que a criação de uma grande tabela de probabilidade baseada em dados históricos. Como exemplo, utilizaremos a base dados do risco de crédito, descrita abaixo:

<p align="center">
  <img src="https://user-images.githubusercontent.com/30511610/83408990-ee5cfb80-a3e9-11ea-8ad4-10e76839c7f2.png" width="70%">
</p>

A tabela de probabilidade objetiva, para cada atributo da tabela, indicar a quantidade de ocorrências em relação ao total de ocorrências em que aquele atributo está associado às classes de risco alto, moderado ou baixo. Vejamos abaixo como fica esta tabela após a etapa de treinamento.

<p align="center">
  <img src="https://user-images.githubusercontent.com/30511610/83409147-39770e80-a3ea-11ea-892d-cd15bbd52d81.png" width="70%">
</p>

Ou seja, analisando primeiramente somente as classes, veremos que 6 de um total de 14 registros do dataset possuem risco de crédito alto, ao passo que apenas 3 registros são considerados de risco moderado e 5 de risco baixo. 
Agora, analisando a relação do atributo história de crédito boa com as classes, veremos que existe apenas 1 registro em um total de 6 em que este atributo está associado a um risco alto. Ao mesmo tempo que existe apenas 1 registro em 3 associado ao risco de crédito moderado, e 3 registros em 5 ao risco baixo. Dessa forma, a etapa de treinamento do algoritmo analisa sucessivamente a relação da ocorrência de cada atributo em relação as classes, construindo a tabela de probabilidade, a qual servirá como base para a predição de novos registros.

A correção Laplaciana é um artifício utilizado para tratar a ocorrência de valores nulos em nossa tabela de probabilidade. A correção consiste em adicionar um registro a mais para substituir o zero e nãop interferir no cálculo final.

## Predições
Agora digamos que no banco em questão, um novo cliente deseja solicitar um empréstimo possuindo os seguintes atributos:

<p align="center">
  <img src="https://user-images.githubusercontent.com/30511610/83411394-86f57a80-a3ee-11ea-9ab0-f61fa952a359.png" width="70%">
</p>

Como calcular a probabilidade do risco de empréstimo para este cliente? A partir da tabela de probabilidade construída na etapa de treinamento é bem simples.

<p align="center">
  <img src="https://user-images.githubusercontent.com/30511610/83411887-7eea0a80-a3ef-11ea-8b3d-7fa3dc415e6d.png" width="70%">
</p>

Na nossa tabela de probabilidade, selecionaremos apenas as colunas referentes aos atributos que o usuário possui para então calcular a probabilidade de cada classe de risco. O cálculo é bem simples:

```
Probabilidade de risco dado os atributos do cliente:

P(Alto) = 6/14 * 1/6 * 4/6 * 6/6 * 1/6 
P(Alto) = 0,0079

P(Moderado) = 3/14 * 1/3 * 1/3 * 2/3 * 1/3 
P(Moderado) = 0,0052

P(Baixo) = 5/14 * 3/5 * 2/5 * 3/5 * 5/5 = 0,0514
P(Baixo) = 0,0514
```

Para obter estes valores em porcentagem é simples:

```
Total = 0,0079 + 0,0052 + 0,0514 = 0,0645 (100%)

P(Alto) = 0,0079/0,0645 * 100 = 12,24%
P(Moderado) = 0,0052/0,0645 * 100 = 8,06%
P(Baixo) = 0,00514/0,0645 * 100 = 79,68%

```

## Vantagens
- Rápido
- Simplicidade de interpretação
- Trabalha com altas dimensões
- Boas previsões em bases pequenas

## Desvantagens
- Combinação de características (atributos independentes) - cada par de características são independentes - nem sempre se aplica no mundo real.

# Naive Bayes com Scikit-Learn
O Scikit-Learn oferece uma classe específica para aplicação do Naive Bayes. Para gerar a tabela de probabilidade, execute:

```
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)
```

É possível utilizar o método `predict` para realizar previsões, como o utilizado no script em que aplica-se NB na base de risco de crédito:

```
previsao = classificador.predict([[0, 0, 1, 2], [3, 0, 0, 0]])
```

A partir da separação do dataset em treinamento e teste, é possível passar a lista previsores_teste para executar a predição e analisar a sua acurácia.

```
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
```

Uma importante análise ocorre através da matriz de confusão, que permite analisar a quantidade de erros e acertos para cada classe.

```
matriz = confusion_matrix(classe_teste, previsoes)
```
Por exemplo, como utilizado no script `naive_bayes_credit_data.py`, ao executar esta linha, você visualizará a seguinte matriz:

<p align="center">
  <img src="https://user-images.githubusercontent.com/30511610/83521959-5aa53100-a4b6-11ea-88fd-2e0fbf0ae07e.png">
</p>

É muito importante interpretar estes dados corretamente: o primeiro valor, de 428, mostra o total de vezes em que a classe 0 foi corretamente classificada como 0 nos testes, ao passo de que o número 8, mostra a quantidade de vezes que a classe zero foi classificada como 1 nos testes. O mesmo ocorre para a segunda linha: o número 23 é o total de vezes que a classe 1 foi classificada como 0, tal como o número 41 consiste no total de acertos para a classe 1. Ou seja, somando 428 + 41, teremos um total de 469 acertos. Essa análise permite observar onde mais está ocorrendo erros e tomar medidas a partir disso. 