import datetime
import warnings
import numpy as np
import pandas as pd
from itertools import product
from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from typing import Dict, List, Union
from xgboost import XGBRegressor

from frameworks_wolfram.wolfram.config import Config


warnings.filterwarnings(action='ignore')


class WrapperEstimator(object):

    def __init__(
            self,
            config: Config,
            estimator_name: str,
    ):
        self._config = config
        self._estimator = _Estimator(estimator_name)

    def get_param_grid(self) -> List[Dict[str, Union[float, int]]]:
        return self._estimator.param_grid

    def set_params(self, params: Dict[str, Union[float, int]]) -> None:
        self._estimator.model.set_params(**params)

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._estimator.model.fit(x_train, y_train)

    def predict_train(self, x_train: pd.DataFrame) -> pd.Series:
        y_train = self._estimator.model.predict(x_train)
        y_train = pd.Series(y_train, x_train.index)
        return y_train

    def predict_test(self, y_train: pd.Series, x_test: pd.DataFrame) -> pd.Series:
        dates_test = x_test.index
        date_last = dates_test[-1]
        y_test = pd.Series(index=dates_test)
        y_stat = pd.concat(objs=[y_train, y_test], axis=0, verify_integrity=True)
        for date in dates_test:
            y_test.loc[date] = y_stat.loc[date] = self._estimator.model.predict(x_test.loc[[date]])[0]
            if date == date_last:
                break
            for ws in self._config.window_sizes:
                param_name = f'{y_train.name}_{ws}'
                window = y_stat.dropna().iloc[-ws:]
                date_next = date + datetime.timedelta(days=1)
                x_test.loc[date_next, f'{param_name}_median'] = window.median()
                for q in self._config.quantiles:
                    x_test.loc[date_next, f'{param_name}_quantile_{q}'] = window.quantile(q)
        return y_test


class _Estimator(object):

    _estimators = {
        'ela': (
            ElasticNet(
                max_iter=1e5,
                random_state=1,
            ),
            {
                'alpha': np.arange(1, 11, 1),
                'l1_ratio': np.arange(0.1, 1.1, 0.1),
            },
        ),
        'svr': (
            LinearSVR(
                max_iter=1e5,
                random_state=1,
            ),
            {
                'C': np.arange(1, 2.1, 1),
            },
        ),
        'xgb': (
            XGBRegressor(
                verbosity=0,
                random_state=1,
            ),
            {
                'n_estimators': np.arange(20, 50, 10),
                'max_depth': [3, 4, 5],
                'reg_alpha': [0.1, 1, 10],
            },
        ),
    }

    def __init__(
            self,
            estimator_name: str,
    ):
        self._estimator_name = estimator_name
        self._set_attributes()

    def _set_attributes(self) -> None:
        model, param_dict = self._estimators[self._estimator_name]
        self._model = clone(model)
        self._param_grid = []
        for params in product(*param_dict.values()):
            self._param_grid.append(dict(zip(param_dict.keys(), params)))

    @property
    def param_grid(self) -> List[Dict[str, Union[float, int]]]:
        return self._param_grid

    @property
    def model(self) -> Union[ElasticNet, LinearSVR, XGBRegressor]:
        return self._model
