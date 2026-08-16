# Classificação Fraudes_com_LogisticRegression

Desenvolvido em Python para aplicação de Machine Learning na classificação de transações de cartão de crédito.
O modelo utiliza **Regressão Logística** para classificar as transações entre legítimas e fraudulentas.

## Tecnologias

* Python
* Pandas
* NumPy
* Scikit-learn

## Dataset

Dataset público de transações de cartão de crédito:

```text id="5f8b0a"
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

## Desenvolvimento

O projeto realiza as seguintes etapas:

* Carregamento e análise dos dados
* Verificação do desbalanceamento das classes
* Feature Engineering
* Transformação logarítmica da variável `Amount`
* Padronização dos dados com `StandardScaler`
* Separação dos dados em treino e teste
* Treinamento do modelo de Regressão Logística
* Classificação das transações

## Execução

Instale as dependências:

```bash id="x9f3ba"
pip install pandas numpy scikit-learn
```

Execute o projeto:

```bash id="f6b2ce"
python main.py
```

O dataset é carregado diretamente da URL durante a execução.

## Conceitos aplicados

* Machine Learning supervisionado
* Classificação binária
* Feature Engineering
* Normalização e padronização
* Regressão Logística
* Tratamento de dados desbalanceados
* Separação de dados para treinamento e teste

## Sobre

Projeto proposto pela plataforma DIO.
Desenvolvido como parte dos meus estudos práticos em **Python, Análise de Dados e Machine Learning**.
