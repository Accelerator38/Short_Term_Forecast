import pickle
from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, debits, pressures, coordinates):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def to_excel(self, file_path):
        pass

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)


class ClusteredBaseModel(BaseModel):

    @abstractmethod
    def get_x_mask(self, cluster_n):
        pass