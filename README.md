# RNA
 estudos de redes neurais artificiais

# instalar o requeriments.txt
````cmd
pip install requeriments.txt
````

# import de lib
````python
import pandas as pd
````

# divisão de classe e previsores

previsores é X

classes é Y

````python
import pandas as pd

base_tabela_processo = pd.read_csv('<CAMINHO>')
````

| id   | npu                  | classe_civil | valor    | moviemnto | tipo_movimeto | tag        | procedente |
|------|----------------------|--------------|----------|-----------|---------------|------------|------------|
| 0152 | 00025587420188262251 | penal        | 33665,24 | 01115     |               | TJSP_civil | 1          |
| 0152 | 00025587420188262251 | penal        | 33665,24 | 01116     |               | TJSP_civil | 1          |
| 0158 | 00035579420188262251 | penal        | 36689,24 | 01185     |               | TJSP_civil | 0          |

## COLUNAS

base_tabela_processo.columns
> id, npu, classe_civil, valor, movimento, tag, procedente

vamos ignorar a coluna id

````python

X_processos = base_tabela_processo.iloc[:,1:6].values
y_processos = base_tabela_processo.iloc[:,7].values
````

nota ** não temos a coluna procedente  

## LABEL

transformar dados strings em numericos
````python

from sklearn.preprocessing import LabelEncoder

label_encoder_teste = LabelEncoder()

teste = label_encoder_teste.fit_transform(X_processos[:,1])

# cada uma das colunas vai receber um encoder logo
# o label encoder do valor não é necessario pois ele já é um valor numerico
# nota ** ver como vai ser a tabela movimento pois temos que pegar tudo

label_encoder_npu = LabelEncoder()
label_encoder_classe_civil = LabelEncoder()
label_encoder_tag = LabelEncoder()

X_processos[:,1] = label_encoder_npu.fit transform(X_processos[:,1])
X_processos[:,2] = label_encoder_classe_civil.fit transform(X_processos[:,2])
X_processos[:,5] = label_encoder_tag.fit transform(X_processos[:,5])

````

depois transformar com o OneHotEncoder

````python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# indices das colunas
onehotencoder_processos = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,2,5])], remainder='passthrough')

# o argumento remainder='passthrough' serve para ele não ignorar os outros indices numericos não listados

X_processos = onehotencoder_processos.fit_transform(X_processos).toarry()

````

# padronização da coluna valores

````python
from sklearn.preprocessing import StandardScaler

scaler_processos = StandardScaler()

X_prcessos = scaler_processos.fit_transform(X_processos)
````

# Divisão da base em treinamento e teste

````python

from sklearn.model_selection import train_test_split

X_treinamento_processsos, X_teste_processos, Y_treinamento_processos, y_teste_processos = train_test_split(X_processos, y_processos, test_size=0.30, random_state=0)

print(X_treinamento_processsos.shape, X_teste_processos.shape)
````

# salvando os dados

````python
import pickle

with open('processos_base.pkl', mode='wb') as f:
    pickle.dump([X_treinamento_processsos, y_treinamento_processsos, X_teste_processos, y_teste_processos], f)


````
