import hashlib
import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DONWLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("./data/datasets", "housing")
HOUSING_URL = DONWLOAD_ROOT + "./data/datasets/housing/housing.tgz"

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
    return hash(np.int64(indentifier)).digest()[-1]<256 * test_ratio

def split_train_test_por_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_:test_set_check(id_, test_ratio, hash))
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

    print(train_set, test_set)