"""
aqui iremos identificar só um valor que é o preço base do automovel conforme algumas especificações
"""

import pandas as pd

base = pd.read_csv('../data/kaggle_car/autos.csv', encoding='ISO-8859-1') # esse encod é pq tem acentos

print(base.shape)

"""
 0   dateCrawled          371528 non-null  object # data_de_extração
 1   name                 371528 non-null  object # nome_veiculo
 2   seller               371528 non-null  object # nome_do_vendedor
 3   offerType            371528 non-null  object # oferta
 4   price                371528 non-null  int64  # preço _ coluna alvo
 5   abtest               371528 non-null  object 
 6   vehicleType          333659 non-null  object
 7   yearOfRegistration   371528 non-null  int64 
 8   gearbox              351319 non-null  object # tipo de cambio
 9   powerPS              371528 non-null  int64  
 10  model                351044 non-null  object
 11  kilometer            371528 non-null  int64 
 12  monthOfRegistration  371528 non-null  int64 
 13  fuelType             338142 non-null  object 
 14  brand                371528 non-null  object
 15  notRepairedDamage    299468 non-null  object 
 16  dateCreated          371528 non-null  object
 17  nrOfPictures         371528 non-null  int64 
 18  postalCode           371528 non-null  int64 # onde ele está no momento
 19  lastSeen             371528 non-null  object
"""
# pequeno_tratamento excluir as colunas que não vamos usar
base = base.drop(['dateCrawled','dateCreated','postalCode','nrOfPictures','lastSeen'], axis=1) # passa uma lista com as colunas que irão ser retiradas
print(base.shape)

# analise da coluna nome
print(base['name'].value_counts())
"""
vamos retirar o nome por enquanto 
Name: name, Length: 233531, dtype: int64
"""
base = base.drop('name', axis=1)
# analise da coluna vendedor
print(base['seller'].value_counts())
"""
privat        371525
gewerblich         3
Name: seller, dtype: int64
dá para ver que temos mais pessoas vendendo seus carros que as concessionarias,
o que pode resultar em uma diferença imensa nos preços
"""
base = base.drop('seller', axis=1)

# analise da coluna vendedor
print(base['offerType'].value_counts())
"""
Gesuch <- carros vindo de leilão
Angebot    371516
Gesuch         12
Name: offerType, dtype: int64
"""
base = base.drop('offerType', axis=1)

# vamos puxar a coluna price menor ou igual a 10
i1 = base.loc[base.price <= 10] # como não faz sentido ter carro com menos de 10 é um dado incorreto

print(base.price.mean()) # puxando a media

# vamos só usar os valores maiore ou igual a 10
base = base.loc[base.price > 10]

# essa base temos valores muito altos e muito baixos então vamos fazer um pequeno corte
base = base.loc[base.price < 350000]

# salvando uma nova base
base.to_csv('../data/kaggle_car/autos_tratado.csv', encoding='ISO-8859-1')