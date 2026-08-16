import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

print('carregando')

df = pd.read_csv(url)


# Verificando uma classificacao desbalanceada.
df['Class'].value_counts(normalize=True) 


## Criando variaveis que ajudam nosso modelo - Feature Engineering -

    ## criando uma variavel transformando a escala para melhor aprendizagem.
df['Amount_log'] = np.log1p(df['Amount']) 


scaler = StandardScaler ()

    ## Transformando valores.
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])

    ##Preparar o dados para treinar o Modelo.

x = df.drop('Class', axis=1)
y = df['Class']


    #Separar modelo e teste.
X_train, X_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.3,random_state=42)


#Modelo de Classificacao // Prever.

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

