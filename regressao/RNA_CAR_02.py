"""
continuação do tratamento valores faltantes
"""

import pandas as pd

base = pd.read_csv('../data/kaggle_car/autos_tratado.csv', encoding='ISO-8859-1') # esse encod é pq tem acentos

# pegando o nome que mais se repete, não é o ideal
base.loc[pd.isnull(base['vehicleType'])]
print(base['vehicleType'].value_counts()) # no caso aqui é limousine

# caixa de macha
base.loc[pd.isnull(base['gearbox'])]
print(base['gearbox'].value_counts()) # no caso aqui é manuell

# modelo
base.loc[pd.isnull(base['model'])]
print(base['model'].value_counts()) # no caso aqui é golf

# combustivel
base.loc[pd.isnull(base['fuelType'])]
print(base['fuelType'].value_counts()) # no caso aqui é bezin = gasolina

# concerto
base.loc[pd.isnull(base['notRepairedDamage'])]
print(base['notRepairedDamage'].value_counts()) # no caso aqui é nein

# aqui são os parametros, quando ele passar e estiver null ele vai por o que estiver configurado
valores = {'vehicleType':'limousine', 'gearbox':'manuell', 'model':'golf', 'fuelType':'bezin', 'notRepairedDamage':'nein'}

# pondo no base
base = base.fillna(value=valores)

base = base.drop('Unnamed: 0', axis=1)
# aqui ele passa por cima
base.to_csv('../data/kaggle_car/autos_tratado_sem_valores_null.csv', encoding='ISO-8859-1', index=False)