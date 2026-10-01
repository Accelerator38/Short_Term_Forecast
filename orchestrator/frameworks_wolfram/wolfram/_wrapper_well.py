import pandas as pd
from sklearn.metrics import mean_absolute_error
from typing import Dict, List, Union

from frameworks_wolfram.wolfram.config import Config
from frameworks_wolfram.wolfram._wrapper_estimator import WrapperEstimator


class WrapperWell(object):

    def __init__(
            self,
            config: Config,
            well_name: int,
            x_train: pd.DataFrame,
            x_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series,
    ):
        self._config = config
        self._well_name = well_name
        self._x_train = x_train
        self._x_test = x_test
        self._y_train_true = y_train
        self._y_test_true = y_test
        self._run()

    def _run(self) -> None:
        wrapper_estimator = WrapperEstimator(
            self._config,
            self._config.estimator_name_well,
        )
        splitter = _Splitter(
            self._x_train,
            self._y_train_true,
            self._config.is_deep_grid_search,
            self._config.forecast_days_number,
        )
        grid_search = _GridSearch(wrapper_estimator, splitter)
        wrapper_estimator.set_params(grid_search.params)
        wrapper_estimator.fit(self._x_train, self._y_train_true)
        self._y_train_pred = wrapper_estimator.predict_train(self._x_train)
        self._y_test_pred = wrapper_estimator.predict_test(self._y_train_true, self._x_test)

    @property
    def well_name(self) -> int:
        return self._well_name

    @property
    def y_train_pred(self) -> pd.Series:
        return self._y_train_pred

    @property
    def y_test_pred(self) -> pd.Series:
        return self._y_test_pred


class _Splitter(object):

    def __init__(
            self,
            x_train: pd.DataFrame,
            y_train: pd.Series,
            is_deep_split: bool,
            fold_samples_number: int,
    ):
        self._x = x_train
        self._y = y_train
        self._is_deep_split = is_deep_split
        self._fold_samples_number = fold_samples_number
        self._total_samples_number = len(self._y.index)
        self._run()

    def _run(self) -> None:
        self._train_test_pairs = []
        if self._is_deep_split:
            self._set_fold_number()
            self._create_train_test_pair()
        else:
            self._add_pair(self._total_samples_number, self._fold_samples_number)
        self._pair_number = len(self._train_test_pairs)

    def _set_fold_number(self) -> None:
        fn = self._total_samples_number / self._fold_samples_number
        self._fn_int = int(fn)
        self._fn_frac = fn - self._fn_int

    def _create_train_test_pair(self) -> None:
        for i in range(3, self._fn_int + 1):
            pair_samples_number = self._fold_samples_number * i
            test_samples_number = pair_samples_number - self._fold_samples_number
            self._add_pair(pair_samples_number, test_samples_number)
        self._check_last_pair_existing()

    def _check_last_pair_existing(self) -> None:
        if self._fn_frac != 0:
            pair_samples_number = self._total_samples_number
            test_samples_number = self._total_samples_number - self._fold_samples_number * self._fn_int
            self._add_pair(pair_samples_number, test_samples_number)

    def _add_pair(self, pair_samples_number: int, test_samples_number: int) -> None:
        train_samples_number = pair_samples_number - test_samples_number
        x_pair = self._x.head(pair_samples_number)
        y_pair = self._y.head(pair_samples_number)
        x_train = x_pair.head(train_samples_number)
        y_train = y_pair.head(train_samples_number)
        x_test = x_pair.tail(test_samples_number)
        y_test = y_pair.tail(test_samples_number)
        self._train_test_pairs.append({
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
        })

    @property
    def train_test_pairs(self) -> List[Dict[str, Union[pd.DataFrame, pd.Series]]]:
        return self._train_test_pairs

    @property
    def pair_number(self) -> int:
        return self._pair_number


class _GridSearch(object):

    def __init__(
            self,
            wrapper_estimator: WrapperEstimator,
            splitter: _Splitter,
    ):
        self._wrapper_estimator = wrapper_estimator
        self._splitter = splitter
        self._run()

    def _run(self) -> None:
        self._calc_grid()
        self._find_params()

    def _calc_grid(self) -> None:
        self._error_params = {}
        for params in self._wrapper_estimator.get_param_grid():
            error = 0
            for pair in self._splitter.train_test_pairs:
                self._wrapper_estimator.set_params(params)
                self._wrapper_estimator.fit(pair['x_train'], pair['y_train'])
                y_pred = self._wrapper_estimator.predict_test(pair['y_train'], pair['x_test'])
                error += mean_absolute_error(pair['y_test'], y_pred)
            error /= self._splitter.pair_number
            self._error_params[error] = params

    def _find_params(self) -> None:
        error_min = min(self._error_params.keys())
        self._params = self._error_params[error_min]
        print(self._params)

    @property
    def params(self) -> Dict[str, Union[float, int]]:
        return self._params
