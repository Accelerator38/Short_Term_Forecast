import numpy as np
import pandas as pd
from pathlib import Path
# from loguru import logger
from frameworks_hybrid_crm_ml.class_Fedot.config import ConfigFedot
from sklearn.preprocessing import MinMaxScaler
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from scipy.ndimage import shift

class CalculatorFedot:
    def __init__(self, config: ConfigFedot):

        self._config = config

        self.scaler_inj = MinMaxScaler(feature_range=(0, 1))
        self.scaler_water = MinMaxScaler(feature_range=(0, 1))
        self.scaler_oil = MinMaxScaler(feature_range=(0, 1))
        self.scaler_gas = MinMaxScaler(feature_range=(0, 1))
        self.scaler_pressure = MinMaxScaler(feature_range=(0, 1))

        self._calculate()

    # создаём таблицу для статистики
    def _create_true_pred_table(self, train_start, train_end, date, wells_prod, data_prod):
        statistic_liquid = pd.DataFrame(index=date, columns=wells_prod)
        statistic_oil = pd.DataFrame(index=date, columns=wells_prod)
        statistic_gas = pd.DataFrame(index=date, columns=wells_prod)

        wells_true_pred = []
        for well in wells_prod:
            wells_true_pred.append(f"{well}_liq_true")
            wells_true_pred.append(f"{well}_liq_pred")
            wells_true_pred.append(f"{well}_oil_true")
            wells_true_pred.append(f"{well}_oil_pred")
            wells_true_pred.append(f"{well}_gas_true")
            wells_true_pred.append(f"{well}_gas_pred")

        date_train = pd.date_range(train_start, train_end, freq='D')
        date_all = date_train.append(date)
        statistic_all = pd.DataFrame(index=date_all)
        for well in wells_prod:
            statistic_all[f"{well}_liq_true"] = data_prod['Дебит жидкости'][well][train_start:train_end]
            statistic_all[f"{well}_liq_pred"] = np.nan
            statistic_all[f"{well}_oil_true"] = data_prod['Дебит нефти'][well][train_start:train_end]
            statistic_all[f"{well}_oil_pred"] = np.nan
            statistic_all[f"{well}_gas_true"] = data_prod['Дебит газа'][well][train_start:train_end]
            statistic_all[f"{well}_gas_pred"] = np.nan

        return statistic_liquid, statistic_oil, statistic_gas, statistic_all, wells_true_pred

    # переписывает фактические значения на "настоящий фактические"
    def _fill_true_values(self, data_fact, data_pred):
        columns_wells = data_pred.columns
        for well in range(0, len(columns_wells), 6):
            w = data_pred.columns[well].split("_")[0]
            data_pred[columns_wells[well]] = data_fact["Дебит жидкости"][w]
            data_pred[columns_wells[well + 2]] = data_fact["Дебит нефти"][w]
            data_pred[columns_wells[well + 4]] = data_fact["Дебит газа"][w]

        return data_pred

    def _parse_data(self, coeff, lags_xlsx=None):

        project_root = Path(__file__).parent.parent.parent
        path_read = project_root / 'tools_preprocessor' / 'data' / f'{self._config.oilfield}'

        xls = pd.ExcelFile(path_read / f'Input_data_CRM_{self._config.oilfield}.xlsx')

        prod_info = pd.read_excel(xls, sheet_name=0, dtype={"Скважина": str})
        inj_info = pd.read_excel(xls, sheet_name=1, dtype={"Скважина": str})
        # index to datetime
        prod_info["Дата"] = pd.to_datetime(prod_info["Дата"])
        inj_info["Дата"] = pd.to_datetime(inj_info["Дата"])

        data_prod = pd.pivot_table(
            prod_info,
            values=["Дебит жидкости", "Дебит нефти", "Дебит газа", "Забойное давление", "Обводненность"],
            index="Дата",
            columns="Скважина",
            dropna=False,
        )[self._config.train_start:self._config.predict_end]

        data_inj = pd.pivot_table(
            inj_info, values="Приемистость", index="Дата", columns="Скважина", dropna=False
        )[self._config.train_start:self._config.predict_end]

        # coefficient matrix
        coeff.index = coeff.index.astype(str)

        # TODO: продумать сортировку скважин по макисмуму нефти по дате
        # sorted coeff table by oil production last days
        asc = data_prod["Дебит нефти"][coeff.columns].loc[self._config.train_start:]
        asc_wells = asc.sum().sort_values(ascending=False).index
        coeff = coeff.reindex(asc_wells, axis=1)
        coeff[coeff == 0.0] = np.nan

        # TODO: обрезать даты когда имеется лаговое окно
        # data_prod = data_prod.loc[self._config.shift_data:]
        # x_indexes = data_prod.index

        # timelags
        if lags_xlsx:
            lags_xlsx = pd.read_excel(Path(path_read, f"{lags_xlsx}.xlsx"), index_col=0)
            lags_xlsx.index = lags_xlsx.index.astype(str)
            lags_xlsx = lags_xlsx.reindex(coeff.columns, axis=1)
            lags_xlsx = lags_xlsx.reindex(coeff.index, axis=0)
            lags_xlsx = np.array(lags_xlsx.T)

        # TODO: Добавить возможность прогноза на будущее. Продлевать забойное давление и приемистость
        # add new data to find pressure. Continue last values
        # data_prod_fill = pd.concat([data_prod.iloc[[len(data_prod)-1]]]*31).set_index(pd.date_range(predict_start, '2022-06-25'))
        # data_inj_fill = pd.concat([data_inj.iloc[[len(data_inj)-1]]]*31).set_index(pd.date_range(predict_start, '2022-06-25'))
        # for well_fill in ppd_pressure:
        #     data_prod_fill.loc[:,('Забойное давление',well_fill)] = data_prod_fill.loc[:,('Забойное давление',well_fill)] + ppd_pressure[well_fill]
        # for well_fill in ppd_injection:
        #     data_inj_fill[well_fill] = data_inj_fill[well_fill] * ppd_injection[well_fill]
        # data_prod = pd.concat((data_prod, data_prod_fill))
        # data_inj = pd.concat((data_inj, data_inj_fill))

        return data_prod, data_inj, coeff, lags_xlsx

    def _interpolate_data(self, data_prod, data_inj):
        # TODO: продумать как заполнять пропуски получше
        # fill nan
        data_prod = data_prod.interpolate(method="linear", limit_direction="forward")
        data_inj = data_inj.interpolate(method="linear", limit_direction="forward")
        data_prod = data_prod.fillna(0)
        data_inj = data_inj.fillna(0)

        return data_prod, data_inj

    def injection_indexes(self, prod_well: int, coefficient: np.array) -> np.array:
        """
        Возвращает значения с пересчитанными приемистостями с учётом коэффициентов взаимовлияния
        args:
            prod_well - добывающая скважина в формате str
            coefficient - матрица коэффициентов взаимовлияния
        return:
            inj_indexes_influence - DataFrame со значениями приемистости
        """
        # список для индексов влияющих скважин
        inj_indexes_influence = []
        # только те коэффициенты которые входят во влияние
        coeff_prod_well = coefficient[prod_well][~np.isnan(coefficient[prod_well])]
        for value in coeff_prod_well:
            (ind,) = np.where(np.isclose(coefficient[prod_well], value))
            if len(ind) > 1:
                for one_ind in ind:
                    inj_indexes_influence.append(int(one_ind))
            else:
                inj_indexes_influence.append(int(ind))

        return inj_indexes_influence

    def _injection_shift(self, prod_well: int,
            inj_ind: np.array,
            injection: np.array,
            coefficient: np.array,
            data_lags: np.array,
            len_forecast: int,
            cut_date: int,
    ) -> np.array:
        """
        Возвращает значения с пересчитанными приемистостями с учётом коэффициентов взаимовлияния
        args:
            prod_well - добывающая скважина в формате str
            injection - DataFrame с приемистостями
            coefficient - матрица коэффициентов взаимовлияния
        return:
            new_injection - DataFrame со значениями приемистости
        """

        inj_shift = np.zeros([1, cut_date-1]).T
        injection_ppd = np.copy(injection)
        for ind in inj_ind:
            if data_lags:
                inj_shift_one_well = shift(
                    injection_ppd[ind], data_lags[prod_well][ind], cval=np.nan
                )
                inj_shift_one_well = (
                        inj_shift_one_well[-cut_date:] * coefficient[prod_well][ind]
                )
            else:
                inj_shift_one_well = injection_ppd[ind] * coefficient[prod_well][ind]
            # scale
            inj_shift_one_well = inj_shift_one_well.reshape((len(inj_shift_one_well), 1))
            inj_shift_one_well = self.scaler_inj.fit_transform(inj_shift_one_well)
            inj_shift = np.hstack((inj_shift, inj_shift_one_well))

        inj_shift = inj_shift.T[1:]
        inj_shift = np.nan_to_num(inj_shift)

        return inj_shift

    def _make_forecast(self,
            pipeline,
            train: InputData,
            predict: InputData,
            train_exog_pressure: InputData,
            predict_exog_pressure: InputData,
            exog_injection,
    ):
        """
        Function for predicting values in a time series
        :return predicted_values: numpy array, forecast of model
        """

        if train_exog_pressure is None:
            second_node_name = "data_source_ts/2"
            train_exog = train
            predict_exog = predict

            train_dataset = MultiModalData(
                {"data_source_ts/1": train, second_node_name: train_exog}
            )

            predict_dataset = MultiModalData(
                {"data_source_ts/1": predict, second_node_name: predict_exog}
            )

            pipeline.fit_from_scratch(train_dataset)

            # Predict
            predicted_values = pipeline.predict(predict_dataset)
            predicted_values = predicted_values.predict

            return predicted_values

        # Fit it
        dict_for_train_dataset = {
            "data_source_ts/1": train,
            "exog_ts/1": train_exog_pressure,
        }
        dict_for_predict_dataset = {
            "data_source_ts/1": predict,
            "exog_ts/1": predict_exog_pressure,
        }
        if len(exog_injection) > 0:
            j = 2
            for i in range(0, len(exog_injection), 2):
                dict_for_train_dataset[f"exog_ts/{j}"] = exog_injection[i]
                dict_for_predict_dataset[f"exog_ts/{j}"] = exog_injection[i + 1]
                j += 1

        train_dataset = MultiModalData(dict_for_train_dataset)

        predict_dataset = MultiModalData(dict_for_predict_dataset)

        pipeline.fit_from_scratch(train_dataset)

        # Predict
        predicted_values = pipeline.predict(predict_dataset)
        predicted_values = predicted_values.predict

        return predicted_values

    def _create_pipeline(self, inj_amount, with_exog=True):
        node_source = PrimaryNode("data_source_ts/1")
        node_lagged = SecondaryNode("lagged", nodes_from=[node_source])
        #     node_lagged.custom_params = {'window_size': 7}

        node_exog_pressure = PrimaryNode("exog_ts/1")
        if inj_amount > 0:
            nodes_exog_injection = [
                PrimaryNode(f"exog_ts/{i + 2}") for i in range(0, inj_amount)
            ]
            nodes_from = [node_lagged, node_exog_pressure, *nodes_exog_injection]
        else:
            nodes_from = [node_lagged, node_exog_pressure]

        node_final = SecondaryNode("ridge", nodes_from=nodes_from)

        # without exog
        if with_exog == False:
            node_source_1 = PrimaryNode("data_source_ts/1")
            node_source_2 = PrimaryNode("data_source_ts/2")

            node_lagged_1 = SecondaryNode("lagged", nodes_from=[node_source_1])
            node_lagged_2 = SecondaryNode("lagged", nodes_from=[node_source_2])

            node_ridge_1 = SecondaryNode("ridge", nodes_from=[node_lagged_1])
            node_ridge_2 = SecondaryNode("ridge", nodes_from=[node_lagged_2])
            node_final = SecondaryNode("ridge", nodes_from=[node_ridge_1, node_ridge_2])
            train_input_exog = None
            predict_input_exog = None
            pipeline = Pipeline(node_final)

        return Pipeline(node_final)

    def _train_test_arrays(self, target, feature, task):
        predict_input_exog = InputData(
            idx=np.arange(len(feature)),
            features=feature,
            target=target,
            task=task,
            data_type=DataTypesEnum.ts,
        )
        train_input_exog, _ = train_test_data_setup(predict_input_exog)
        return train_input_exog, predict_input_exog

    def _run(self,
            data_water: np.array,
            data_oil: np.array,
            data_gas: np.array,
            data_pressure,
            data_injection,
            coeff,
            lags,
            cut_date,
            len_forecast=31,
            with_exog=True,
    ):
        """Function with example how time series forecasting can be made with using
        exogenous features
        :param path_to_file: path to the csv file with dataframe
        :param len_forecast: forecast length
        :param with_exog: is it needed to make prediction with exogenous time series
        :param with_visualisation: is it needed to make visualisations
        """
        # matrix for statistics
        statistics = np.zeros(
            [
                len_forecast,
            ]
        )
        statistics_liq = np.zeros(
            [
                len_forecast,
            ]
        )
        statistics_oil = np.zeros(
            [
                len_forecast,
            ]
        )
        statistics_gas = np.zeros(
            [
                len_forecast,
            ]
        )
        for prod_well in range(coeff.shape[0]):
            #         print(f'{prod_well}/{coeff.shape[0]}')
            time_series_water_fact = data_water[prod_well]
            time_series_oil_fact = data_oil[prod_well]
            time_series_gas_fact = data_gas[prod_well]
            # ppd pressure
            pressure = np.copy(data_pressure[prod_well])
            # scale
            time_series_water = time_series_water_fact.reshape(
                (len(time_series_water_fact), 1)
            )
            time_series_oil = time_series_oil_fact.reshape(
                (len(time_series_oil_fact), 1)
            )
            time_series_gas = time_series_gas_fact.reshape(
                (len(time_series_gas_fact), 1)
            )
            pressure = pressure.reshape((len(pressure), 1))
            time_series_water = self.scaler_water.fit_transform(time_series_water)
            time_series_oil = self.scaler_oil.fit_transform(time_series_oil)
            time_series_gas = self.scaler_gas.fit_transform(time_series_gas)
            pressure = self.scaler_pressure.fit_transform(pressure)
            # reshape to predict
            time_series_water = time_series_water.T[0]
            time_series_oil = time_series_oil.T[0]
            time_series_gas = time_series_gas.T[0]
            # injection index
            influence_inj_indexes = self.injection_indexes(prod_well, coeff)

            if len(influence_inj_indexes) >= 1:
                # shift injection
                inj_to_feature_water_shift = self._injection_shift(
                    prod_well,
                    influence_inj_indexes,
                    data_injection,
                    coeff,
                    lags,
                    len_forecast,
                    cut_date,
                )
                # prepare exog_variable
                exog_variable = pressure
                exog_variable_injection = inj_to_feature_water_shift
            else:
                exog_variable = pressure
                exog_variable_injection = None

            # Source time series
            train_input_water, predict_input_water = train_test_data_setup(
                InputData(
                    idx=range(len(time_series_water)),
                    features=time_series_water,
                    target=time_series_water,
                    task=Task(
                        TaskTypesEnum.ts_forecasting,
                        TsForecastingParams(forecast_length=len_forecast),
                    ),
                    data_type=DataTypesEnum.ts,
                )
            )
            train_input_oil, predict_input_oil = train_test_data_setup(
                InputData(
                    idx=range(len(time_series_oil)),
                    features=time_series_oil,
                    target=time_series_oil,
                    task=Task(
                        TaskTypesEnum.ts_forecasting,
                        TsForecastingParams(forecast_length=len_forecast),
                    ),
                    data_type=DataTypesEnum.ts,
                )
            )
            train_input_gas, predict_input_gas = train_test_data_setup(
                InputData(
                    idx=range(len(time_series_gas)),
                    features=time_series_gas,
                    target=time_series_gas,
                    task=Task(
                        TaskTypesEnum.ts_forecasting,
                        TsForecastingParams(forecast_length=len_forecast),
                    ),
                    data_type=DataTypesEnum.ts,
                )
            )

            # Exogenous time series
            task_water = Task(
                TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast),
            )
            task_oil = Task(
                TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast),
            )
            task_gas = Task(
                TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast),
            )
            train_input_exog_water, predict_input_exog_water = self._train_test_arrays(
                time_series_water, exog_variable, task_water
            )
            train_input_exog_oil, predict_input_exog_oil = self._train_test_arrays(
                time_series_oil, exog_variable, task_oil
            )
            train_input_exog_gas, predict_input_exog_gas = self._train_test_arrays(
                time_series_gas, exog_variable, task_gas
            )
            exog_injection_water = []
            exog_injection_oil = []
            exog_injection_gas = []
            for inj_index in range(len(influence_inj_indexes)):
                (
                    train_input_exog_water_injection,
                    predict_input_exog_water_injection,
                ) = self._train_test_arrays(
                    time_series_water, exog_variable_injection[inj_index], task_water
                )
                (
                    train_input_exog_oil_injection,
                    predict_input_exog_oil_injection,
                ) = self._train_test_arrays(
                    time_series_oil, exog_variable_injection[inj_index], task_oil
                )
                exog_injection_water.append(train_input_exog_water_injection)
                exog_injection_water.append(predict_input_exog_water_injection)
                exog_injection_oil.append(train_input_exog_oil_injection)
                exog_injection_oil.append(predict_input_exog_oil_injection)

            # Example with exogenous time series
            pipeline_water = self._create_pipeline(
                len(influence_inj_indexes), with_exog
            )
            pipeline_oil = self._create_pipeline(len(influence_inj_indexes), with_exog)

            pipeline_gas = self._create_pipeline(0, with_exog)
            # for exog
            if with_exog == False:
                node_source_1 = PrimaryNode("data_source_ts/1")
                node_source_2 = PrimaryNode("data_source_ts/2")

                node_lagged_1 = SecondaryNode("lagged", nodes_from=[node_source_1])
                node_lagged_2 = SecondaryNode("lagged", nodes_from=[node_source_2])

                node_ridge_1 = SecondaryNode("ridge", nodes_from=[node_lagged_1])
                node_ridge_2 = SecondaryNode("ridge", nodes_from=[node_lagged_2])
                node_final = SecondaryNode(
                    "ridge", nodes_from=[node_ridge_1, node_ridge_2]
                )
                train_input_exog_water = None
                predict_input_exog_water = None
                train_input_exog_oil = None
                predict_input_exog_oil = None
                train_input_exog_gas = None
                predict_input_exog_gas = None
                exog_injection_water = None
                exog_injection_oil = None
                pipeline_water = Pipeline(node_final)
                pipeline_oil = Pipeline(node_final)
                pipeline_gas = Pipeline(node_final)

            predicted_water = self._make_forecast(
                pipeline_water,
                train_input_water,
                predict_input_water,
                train_input_exog_water,
                predict_input_exog_water,
                exog_injection_water,
            )
            predicted_oil = self._make_forecast(
                pipeline_oil,
                train_input_oil,
                predict_input_oil,
                train_input_exog_oil,
                predict_input_exog_oil,
                exog_injection_oil,
            )
            predicted_gas = self._make_forecast(
                pipeline_gas,
                train_input_gas,
                predict_input_gas,
                train_input_exog_gas,
                predict_input_exog_gas,
                exog_injection_gas,
            )

            predicted_water = np.ravel(np.array(predicted_water))
            test_data_water = np.ravel(predict_input_water.target)

            predicted_oil = np.ravel(np.array(predicted_oil))
            test_data_oil = np.ravel(predict_input_oil.target)

            predicted_gas = np.ravel(np.array(predicted_gas))
            test_data_gas = np.ravel(predict_input_gas.target)

            # rescale
            predicted_water = predicted_water.reshape((len(predicted_water), 1))
            test_data_water = test_data_water.reshape((len(test_data_water), 1))
            predicted_oil = predicted_oil.reshape((len(predicted_oil), 1))
            test_data_oil = test_data_oil.reshape((len(test_data_oil), 1))
            predicted_gas = predicted_gas.reshape((len(predicted_gas), 1))
            test_data_gas = test_data_gas.reshape((len(test_data_gas), 1))

            predicted_water = self.scaler_water.inverse_transform(predicted_water)
            test_data_water = self.scaler_water.inverse_transform(test_data_water)
            predicted_oil = self.scaler_oil.inverse_transform(predicted_oil)
            test_data_oil = self.scaler_oil.inverse_transform(test_data_oil)
            predicted_gas = self.scaler_gas.inverse_transform(predicted_gas)
            test_data_gas = self.scaler_gas.inverse_transform(test_data_gas)

            predicted_water = predicted_water.T[0]
            test_data_water = test_data_water.T[0]
            predicted_oil = predicted_oil.T[0]
            test_data_oil = test_data_oil.T[0]
            predicted_gas = predicted_gas.T[0]
            test_data_gas = test_data_gas.T[0]

            statistics = np.vstack(
                (
                    statistics,
                    test_data_water,
                    predicted_water,
                    test_data_oil,
                    predicted_oil,
                    test_data_gas,
                    predicted_gas,
                )
            )
            statistics_liq = np.vstack((statistics_liq, predicted_water))
            statistics_oil = np.vstack((statistics_oil, predicted_oil))
            statistics_gas = np.vstack((statistics_gas, predicted_gas))

        return statistics[1:], statistics_liq[1:], statistics_oil[1:], statistics_gas[1:]


    def _calculate(self):
        # logger.info("Fedot calculation: START")
        data_prod_fact, data_inj_fact, coeff, data_lags = self._parse_data(
            self._config.coeff_f, self._config.lags
        )
        # only work wells
        coeff = coeff[set(self._config.wells_norm) & set(coeff.columns)]
        data_prod, data_inj = self._interpolate_data(data_prod_fact, data_inj_fact)
        # check data_inj's first is train_start
        if data_inj.index[0] != pd.Timestamp(self._config.train_start):
            add_date = pd.date_range(self._config.train_start, data_inj.index[0], freq='D', inclusive="left")
            empty_df = pd.DataFrame(data=0, index=add_date, columns=data_inj.columns)
            data_inj = pd.concat([empty_df, data_inj], ignore_index=False)

        # для массивов определяем длину. Добавляем "1" для совпадения по длине
        if data_lags:
            shift_date = (self._config.predict_end - self._config.train_start).days - data_lags.max().max() + 1
        else:
            shift_date = self._config.train_start

        (
            statistic_liquid,
            statistic_oil,
            statistic_gas,
            self.statistic_all,
            wells_true_pred,
        ) = self._create_true_pred_table(self._config.train_start, self._config.train_end,
                                         self._config.dates_predict, coeff.columns, data_prod)

        last_predict = len(self._config.dates_predict) % self._config.date_discr
        for end_cut in range(self._config.date_discr - 1, len(self._config.dates_predict) + self._config.date_discr, self._config.date_discr):
            print(f"Number of iteration: {int(end_cut / self._config.date_discr)}/{int(len(self._config.dates_predict) / self._config.date_discr)}")
            if (end_cut + 1 == len(self._config.dates_predict) + self._config.date_discr) and (last_predict == 0):
                break
            # if there is a non-integer number of days remaining
            if len(self._config.dates_predict) + self._config.date_discr - end_cut == last_predict + 1:
                predict_dates_day = pd.date_range(
                    self._config.dates_predict[end_cut - self._config.date_discr + 1],
                    self._config.dates_predict[end_cut - self._config.date_discr + last_predict],
                    freq="D",
                )
                end_cut = end_cut - self._config.date_discr + last_predict
                self._config.date_discr = last_predict
            else:
                predict_dates_day = pd.date_range(
                    self._config.dates_predict[end_cut - self._config.date_discr + 1],
                    self._config.dates_predict[end_cut],
                    freq="D",
                )
            # cut date
            cut_date = (
                               pd.to_datetime(self._config.dates_predict[end_cut]) - pd.to_datetime(shift_date)
                       ).days + 1
            # create arrays for model
            data_water = np.array(
                data_prod["Дебит жидкости"][coeff.columns].loc[: self._config.dates_predict[end_cut]].T
            )
            data_oil = np.array(
                data_prod["Дебит нефти"][coeff.columns].loc[: self._config.dates_predict[end_cut]].T
            )
            data_gas = np.array(
                data_prod["Дебит газа"][coeff.columns].loc[: self._config.dates_predict[end_cut]].T
            )
            data_pressure = np.array(
                data_prod["Забойное давление"][coeff.columns]
                    .loc[: self._config.dates_predict[end_cut]]
                    .T
            )
            data_injection = np.array(data_inj[coeff.index].loc[: self._config.dates_predict[end_cut]].T)

            # timeforecast
            (
                prediction_statistics,
                prediction_statistics_liq,
                prediction_statistics_oil,
                prediction_statistics_gas,
            ) = self._run(
                data_water,
                data_oil,
                data_gas,
                data_pressure,
                data_injection,
                np.array(coeff.T),
                data_lags,
                cut_date,
                len_forecast=self._config.date_discr,
                with_exog=True,
            )
            # save liq, oil, gas
            stat_liq = pd.DataFrame(
                data=prediction_statistics_liq.T,
                index=predict_dates_day,
                columns=coeff.columns,
            )
            stat_oil = pd.DataFrame(
                data=prediction_statistics_oil.T,
                index=predict_dates_day,
                columns=coeff.columns,
            )
            stat_gas = pd.DataFrame(
                data=prediction_statistics_gas.T,
                index=predict_dates_day,
                columns=coeff.columns,
            )
            statistic_liquid.loc[predict_dates_day] = stat_liq.loc[predict_dates_day]
            statistic_oil.loc[predict_dates_day] = stat_oil.loc[predict_dates_day]
            statistic_gas.loc[predict_dates_day] = stat_gas.loc[predict_dates_day]

            data_prod["Дебит жидкости"].loc[
                predict_dates_day, (coeff.columns)
            ] = statistic_liquid.loc[predict_dates_day, (coeff.columns)].values
            data_prod["Дебит нефти"].loc[
                predict_dates_day, (coeff.columns)
            ] = statistic_oil.loc[predict_dates_day, (coeff.columns)].values
            data_prod["Дебит газа"].loc[
                predict_dates_day, (coeff.columns)
            ] = statistic_gas.loc[predict_dates_day, (coeff.columns)].values
            # save true, pred
            stat = pd.DataFrame(
                data=prediction_statistics.T,
                index=predict_dates_day,
                columns=wells_true_pred,
            )
            self.statistic_all.loc[predict_dates_day] = stat.loc[predict_dates_day]

        self.statistic_all = self._fill_true_values(data_prod_fact, self.statistic_all)
        # logger.success('Fedot calculation: FINISH}')
        print('CRM+ML calculation: FINISH')
