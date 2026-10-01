from copy import deepcopy
from typing import Dict, List, Tuple, Union

import pandas as pd
from loguru import logger

from frameworks_wolfram.wolfram._wrapper_estimator import WrapperEstimator
from frameworks_wolfram.wolfram._wrapper_well import WrapperWell
from frameworks_wolfram.wolfram.config import Config
from frameworks_wolfram.wolfram.well import Well
from frameworks_wolfram.wolfram.well import WellResults


class Calculator(object):

    def __init__(
            self,
            config: Config,
            wells: List[Well],
    ):
        self._config = config
        self._wells_input = wells
        self._run()

    def _run(self) -> None:
        self._compute()
        self._set_wells_output()

    def _compute(self) -> None:
        self._calculator_rate_liq = _CalculatorRate(
            self._config,
            deepcopy(self._wells_input),  # Расчет по копии
            name_rate_to_predict=Well.NAME_RATE_LIQ,
            name_rate_to_drop_1=Well.NAME_RATE_OIL,
            name_rate_to_drop_2=Well.NAME_RATE_GAS,
        )
        self._calculator_rate_oil = _CalculatorRate(
            self._config,
            deepcopy(self._wells_input),  # Расчет по копии
            name_rate_to_predict=Well.NAME_RATE_OIL,
            name_rate_to_drop_1=Well.NAME_RATE_LIQ,
            name_rate_to_drop_2=Well.NAME_RATE_GAS,
        )
        self._calculator_rate_gas = _CalculatorRate(
            self._config,
            deepcopy(self._wells_input),  # Расчет по копии
            name_rate_to_predict=Well.NAME_RATE_GAS,
            name_rate_to_drop_1=Well.NAME_RATE_OIL,
            name_rate_to_drop_2=Well.NAME_RATE_LIQ,

        )

    def _set_wells_output(self) -> None:
        self._wells_output = []
        wrapper_wells_liq = self._convert_list_to_dict(self._calculator_rate_liq.wrapper_wells)
        wrapper_wells_oil = self._convert_list_to_dict(self._calculator_rate_oil.wrapper_wells)
        wrapper_wells_gas = self._convert_list_to_dict(self._calculator_rate_gas.wrapper_wells)
        wells_input = self._convert_list_to_dict(self._wells_input)
        well_names = sorted(set(wrapper_wells_liq.keys()) & set(wrapper_wells_oil.keys()) & set(wrapper_wells_gas.keys()))
        for well_name in well_names:
            well_results = WellResults(
                rates_liq_train=wrapper_wells_liq[well_name].y_train_pred,
                rates_liq_test=wrapper_wells_liq[well_name].y_test_pred,
                rates_oil_train=wrapper_wells_oil[well_name].y_train_pred,
                rates_oil_test=wrapper_wells_oil[well_name].y_test_pred,
                rates_gas_train=wrapper_wells_gas[well_name].y_train_pred,
                rates_gas_test=wrapper_wells_gas[well_name].y_test_pred,
            )
            well = wells_input[well_name]
            well.results = well_results
            self._wells_output.append(well)

    @staticmethod
    def _convert_list_to_dict(items_list: List[Union[WrapperWell, Well]]) -> Dict[int, Union[WrapperWell, Well]]:
        keys = [item.well_name for item in items_list]
        items_dict = dict(zip(keys, items_list))
        return items_dict

    @property
    def wells(self) -> List[Well]:
        return self._wells_output


class _CalculatorRate(object):

    def __init__(
            self,
            config: Config,
            wells: List[Well],
            name_rate_to_predict: str,
            name_rate_to_drop_1: str,
            name_rate_to_drop_2: str,
    ):
        self._config = config
        self._wells = wells
        self._name_rate_to_predict = name_rate_to_predict
        self._name_rate_to_drop_1 = name_rate_to_drop_1
        self._name_rate_to_drop_2 = name_rate_to_drop_2
        self._run()

    def _run(self) -> None:
        self._handle()
        self._compute()

    def _handle(self) -> None:
        x, y = [], []
        self._well_data = {}
        for well in self._wells:
            handler = _Handler(
                self._config,
                self._name_rate_to_predict,
                self._name_rate_to_drop_1,
                self._name_rate_to_drop_2,
                well.df,
            )
            data = handler.data
            train_days_number = len(data['y_train'])
            if train_days_number < self._config.forecast_days_number:
                text = f'Скважина {well.well_name} не будет рассчитана по параметру "{self._name_rate_to_predict}",'
                f'так как длительность периода адаптации меньше, чем периода прогнозирования.'
                logger.error(text)
                print(text)
                continue
            x.append(data['x_train'])
            y.append(data['y_train'])
            self._well_data[well.well_name] = data
        self._x_train = pd.concat(objs=x, ignore_index=True)
        self._y_train = pd.concat(objs=y, ignore_index=True)
        self._create_group_estimator()

    def _create_group_estimator(self) -> None:
        self._group_estimator = WrapperEstimator(
            self._config,
            self._config.estimator_name_group,
        )
        self._group_estimator.fit(self._x_train, self._y_train)

    def _compute(self) -> None:
        self._wrapper_wells = []
        for well_name, data in self._well_data.items():
            print(well_name)
            logger.info(f'Wolfram: well {well_name}')
            try:
                x_train, y_train, x_test, y_test = data.values()
                x_train[Well.NAME_RATE_BASE] = self._group_estimator.predict_train(x_train)
                x_test[Well.NAME_RATE_BASE] = self._group_estimator.predict_test(y_train, x_test)
                wrapper_well = WrapperWell(
                    self._config,
                    well_name,
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                )
                self._wrapper_wells.append(wrapper_well)
                logger.success(f'Wolfram: success {well_name}')
            except Exception as exc:
                logger.exception(f'Wolfram: FAIL {well_name}', exc)
                # continue

    @property
    def wrapper_wells(self) -> List[WrapperWell]:
        return self._wrapper_wells


class _Handler(object):

    def __init__(
            self,
            config: Config,
            rate_name_to_predict: str,
            rate_name_to_drop_1: str,
            rate_name_to_drop_2: str,
            df: pd.DataFrame,
    ):
        self._config = config
        self._rate_name_to_predict = rate_name_to_predict
        self._rate_name_to_drop_1 = rate_name_to_drop_1
        self._rate_name_to_drop_2 = rate_name_to_drop_2
        self._df = df
        self._run()

    def _run(self) -> None:
        self._data = {}
        self._drop_excess_rate()
        self._interpolate_gaps()
        self._add_new_features()
        self._split_train_test_x_y()

    def _drop_excess_rate(self) -> None:
        self._df.drop(columns=[self._rate_name_to_drop_1, self._rate_name_to_drop_2], inplace=True)

    def _interpolate_gaps(self) -> None:
        index = self._df.index
        self._df.reset_index(drop=True, inplace=True)
        self._df.interpolate(method='ffill', inplace=True)
        self._df.interpolate(method='bfill', inplace=True)
        self._df.index = index

    def _add_new_features(self) -> None:
        for col in self._df.columns:
            for ws in self._config.window_sizes:
                self._create_window_features(col, ws)
        self._df.dropna(inplace=True)

    def _create_window_features(self, col: str, ws: int) -> None:
        window = self._df[col].rolling(ws)
        col += f'_{ws}'
        self._df[f'{col}_median'] = window.median().shift()
        for q in self._config.quantiles:
            self._df[f'{col}_quantile_{q}'] = window.quantile(q).shift()

    def _split_train_test_x_y(self) -> None:
        df_train = self._df.head(len(self._df.index) - self._config.forecast_days_number)
        df_test = self._df.tail(self._config.forecast_days_number)
        self._data['x_train'], self._data['y_train'] = self._divide_x_y(df_train)
        self._data['x_test'], self._data['y_test'] = self._divide_x_y(df_test)

    def _divide_x_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        x = df.drop(columns=self._rate_name_to_predict)
        y = df[self._rate_name_to_predict]
        return x, y

    @property
    def data(self) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        return self._data
