import hashlib
import os
import tarfile
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder

from util.CominedAtributesAdder import *


DONWLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("../data/datasets", "housing")
HOUSING_URL = DONWLOAD_ROOT + "../data/datasets/housing/housing.tgz"

def fecth_housing(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

    tgz_path = os.path.join(housing_path, "housing.tgz")

    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# separação dos dados
def split_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    teste_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:teste_set_size]
    train_indices = shuffled_indices[teste_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#indentificação dos dados de testes
def test_set_check(indentifier, test_ratio, hash):
    return hash(np.int64(indentifier)).digest()[-1] < 256 * test_ratio

def split_train_test_por_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


if __name__ == "__main__":
    # fecth_housing()
    housing = load_housing_data()
    # print(housing.head())
    # print(housing.info())
    ocean_proximity = housing["ocean_proximity"].value_counts()
    # print(ocean_proximity)
    # print(housing.describe())

    # housing.hist(bins=50, figsize=(20,15))
    # plt.show()

    housing_with_id = housing.reset_index() #add uma coluna indice
    train_set, test_set = split_train_test_por_id(housing_with_id, 0.2, "index")
    # para criar id unicos usar valores estaveis algo inmutavel
    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    train_set, test_set = split_train_test_por_id(housing_with_id, 0.2, "id")

    # print(train_set, test_set)

    housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    #
    # housing["income_cat"].hist(figsize=(5,5)) #o figsize define as dimeções 5 * 100 pixel
    # plt.show()

    #amostragem estratificada
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
        # print(set_)

    housing = strat_train_set.copy()
    housing.plot(kind='scatter', x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"]/100, label="population", figsize=(10,8),
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
                 )
    plt.legend()
    plt.show()

    ## novos atributos
    housing['rooms_per_household'] = housing['total_rooms']/housing['households']
    housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
    housing['population_per_household'] = housing['population']/housing['households']

    corr_matrix = housing.corr()
    # print(corr_matrix)

    # limpando os dados
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set['median_house_value'].copy()

    median = housing['total_bedrooms'].median()
    housing['total_bedrooms'].fillna(median, inplace=True)

    # manipulando texto e categoria para numeros -> encoder

    #vizualizando as categorias
    housing_cat = housing['ocean_proximity']
    print(housing_cat.head(10))

    #encoder
    housing_cat_encoder, housing_categories = housing_cat.factorize()
    print('encoder',housing_cat_encoder[:10])
    print('categorias', housing_categories)

    #encoder do sklearn
    encoder = OneHotEncoder()
    housing_cat_1hot = encoder.fit_transform(housing_cat_encoder.reshape(-1, 1))

    # transformando em uma matriz como um histograma
    housing_cat_1hot.toarray()
    print('encoder do sklearn',housing_cat_1hot.toarray())


    # attr_adder = CominedAtributesAdder(add_bedrooms_per_room=False)
    # housing_extra_attribs = attr_adder.transform(housing.values)
    #
    # print(housing_extra_attribs)
    # # o profile report só funciona com o pandas 1.2.5, então atenção em qual pandas está instalado
    # profile = ProfileReport(housing, title="Pandas Profiling Report")
    #
    # # o profile gera um arquivo descritivo dos dados
    # profile.to_file("data/housing.html")

    #buscando correlações
    corr_median_house_value = corr_matrix['median_house_value'].sort_values(ascending=False)
    print(corr_median_house_value)