import streamlit as st
import datetime
import pandas as pd
import numpy as np
from tools_preprocessor.config import Config as ConfigPreprocessor
from tools_preprocessor.preprocessor import Preprocessor
from frameworks_shelf_algo.class_Shelf.config import ConfigShelf
from frameworks_shelf_algo.class_Shelf.constants import *
from frameworks_shelf_algo.class_Shelf.support_functions import transform_str_dates_to_datetime_or_vice_versa, \
    datetime_to_str, _get_path

"""Автоматический расчёт темпов падения на основе аппроксимации ретроспективных данных по дебиту.

 На вход: месторождение, цеха, дата начала и окончания прогноза, sh_sost_fond.feather для ГТМ.
 По умолчанию период для расчёта темпов падения 30 дней и 5 дней на осреднение данных.

 Выход: два словаря (нефть, жидкость) с последним значением дебита, темпом падения и данными по ГТМ;
  две таблицы (нефть, жидкость) с фактическими данными на период прогноза"""


# таблицы с фактическими данными могут быть лишними

class DataProcessorShelf:

    def __init__(self, config: ConfigShelf):
        # if st.session_state['change_gtm_info'] == 0:
        #     print("DataProcessorShelf")
        self._config = config
        self.df_fact_test_prd = pd.DataFrame()
        self.data_shelf = dict()
        self.df_fact_test_prd_liq = pd.DataFrame()
        self.df_fact_test_prd_gas = pd.DataFrame()
        self.data_shelf_liq = dict()
        self._path = _get_path(config.oilfield)
        self._run()

    # даты ГТМ
    @staticmethod
    def get_stop_prd_dates(_df_well) -> list[tuple[datetime.date, datetime.date]]:
        col_date = 'date'
        df_statuses = _df_well[[STATUS]].copy()
        df_statuses[col_date] = _df_well.index
        df_statuses.reset_index(drop=True, inplace=True)
        n_rows = len(df_statuses)
        work_dates = []
        i = 0
        while i < n_rows:
            #  Ищем start_date
            start_date = None
            while True:
                if i == n_rows:
                    break
                elif df_statuses.loc[i, STATUS] == 'Остановлена':
                    start_date = df_statuses.loc[i, col_date]
                    i += 1
                    break
                else:
                    i += 1
            # Ищем end_date
            if start_date is not None:
                while True:
                    if i == n_rows:
                        end_date = df_statuses.loc[i - 1, col_date]
                        work_dates.append((start_date, end_date))
                        break
                    elif df_statuses.loc[i, STATUS] == 'В работе':
                        end_date = df_statuses.loc[i - 1, col_date]
                        work_dates.append((start_date, end_date))
                        i += 1
                        break
                    else:
                        i += 1
        return work_dates

    @staticmethod
    def _ensure_gas_defaults(data_shelf: dict) -> None:
        for well, well_data in data_shelf.items():
            if well == PLANNED_MLSP_STOPS:
                continue
            if LAST_MEASUREMENT in well_data:
                well_data[LAST_MEASUREMENT].setdefault(VALUE_GAS, 0.0)
            if DEC_RATES_GAS not in well_data:
                oil_decline_rates = well_data.get(DEC_RATES, {})
                first_date = next(iter(oil_decline_rates), None)
                well_data[DEC_RATES_GAS] = {first_date: 0.0} if first_date is not None else {}

    def _run(self):
        field_name = self._config.oilfield
        shops = self._config.shops
        well_ois = self._config.well_ois
        # print("Data Processor Shelf")
        # print(well_ois)
        n_days_past = self._config.n_days_past
        n_days_calc_avg = self._config.n_days_calc_avg
        date_test = self._config.train_end
        date_end = self._config.predict_end
        date_train = self._config.train_start
        df_fact_test_prd = self.df_fact_test_prd
        data_shelf = self.data_shelf
        df_fact_test_prd_liq = self.df_fact_test_prd_liq
        df_fact_test_prd_gas = self.df_fact_test_prd_gas

        # проверить, в работе ли скважина
        path = self._path / 'sh_sost_fond.feather'
        df_sh_sost_fond = pd.read_feather(path)
        df_sh_sost_fond.set_index('dt', inplace=True)
        st.session_state.first_date = df_sh_sost_fond.first_valid_index()
        date_start = date_train

        preprocessor = Preprocessor(
            ConfigPreprocessor(
                field_name,
                shops,
                date_start,
                date_test,
                date_end,
            )
        )
       # создаём словарь с замерами, ТП и ГТМ для каждой скважины
        data_shelf[MLSP_STOP] = dict()
        for well in well_ois: #preprocessor.well_names:
            # if well in well_ois:
            df_well = df_sh_sost_fond[df_sh_sost_fond['well.ois'] == well]
            if DEBIT_GAS in df_well.columns:
                df_well_gas = df_well[DEBIT_GAS]
            else:
                df_well_gas = pd.Series(0.0, index=df_well.index)
            df_well_work_before_test = df_well[(df_well[STATUS] == 'В работе') & (df_well.index < date_test)]
            df_well_work_before_test_gas = df_well_gas.loc[df_well_work_before_test.index]
            # avg_debit_in_past = np.mean(df_well_work_before_test[DEBIT][-n_days_past:-n_days_past + n_days_calc_avg])
            # avg_debit_in_past_liq = np.mean(
            #     df_well_work_before_test[DEBIT1][-n_days_past:-n_days_past + n_days_calc_avg])
            avg_debit_before_test = np.mean(df_well_work_before_test[DEBIT][-n_days_calc_avg:])
            avg_debit_before_test_liq = np.mean(df_well_work_before_test[DEBIT1][-n_days_calc_avg:])
            avg_debit_before_test_gas = np.mean(df_well_work_before_test_gas[-n_days_calc_avg:])
            # dec_rate = (avg_debit_in_past - avg_debit_before_test) / (n_days_past - n_days_calc_avg)
            # dec_rate_liq = (avg_debit_in_past_liq - avg_debit_before_test_liq) / (n_days_past - n_days_calc_avg)
            # if ((dec_rate != dec_rate) or (dec_rate_liq != dec_rate_liq)):
            if ((avg_debit_before_test != avg_debit_before_test) or
                    (avg_debit_before_test_liq != avg_debit_before_test_liq)):
                continue
                # print(f"{well} is NOT ok")
            else:
                avg_debit_in_past = np.mean(
                    df_well_work_before_test[DEBIT][-n_days_past:-n_days_past + n_days_calc_avg])
                avg_debit_in_past_liq = np.mean(
                    df_well_work_before_test[DEBIT1][-n_days_past:-n_days_past + n_days_calc_avg])
                avg_debit_in_past_gas = np.mean(
                    df_well_work_before_test_gas[-n_days_past:-n_days_past + n_days_calc_avg])
                if ((avg_debit_in_past != avg_debit_in_past) or (avg_debit_in_past_liq != avg_debit_in_past_liq)):
                    first_valid_index_ = df_well_work_before_test.first_valid_index()
                    first_valid_index_avg = first_valid_index_ + datetime.timedelta(days=n_days_calc_avg)
                    avg_debit_in_past = np.mean(
                        df_well_work_before_test[DEBIT][first_valid_index_:first_valid_index_avg])
                    avg_debit_in_past_liq = np.mean(
                        df_well_work_before_test[DEBIT1][first_valid_index_:first_valid_index_avg])
                    avg_debit_in_past_gas = np.mean(
                        df_well_work_before_test_gas[first_valid_index_:first_valid_index_avg])
                if avg_debit_before_test_gas != avg_debit_before_test_gas:
                    avg_debit_before_test_gas = 0.0
                if avg_debit_in_past_gas != avg_debit_in_past_gas:
                    avg_debit_in_past_gas = avg_debit_before_test_gas
                dec_rate = (avg_debit_in_past - avg_debit_before_test) / (n_days_past - n_days_calc_avg)
                dec_rate_liq = (avg_debit_in_past_liq - avg_debit_before_test_liq) / (n_days_past - n_days_calc_avg)
                dec_rate_gas = (avg_debit_in_past_gas - avg_debit_before_test_gas) / (n_days_past - n_days_calc_avg)
                if ((dec_rate != dec_rate) or (dec_rate_liq != dec_rate_liq)):
                    # print(f"{well} /GFB. is NOT ok, dec_rate")
                    continue
                else:
                    df_fact_test_prd[well] = df_well[DEBIT][(date_train <= df_well.index) & (df_well.index <= date_end)]
                    df_fact_test_prd_liq[well] = df_well[DEBIT1][
                        (date_train <= df_well.index) & (df_well.index <= date_end)]
                    df_fact_test_prd_gas[well] = df_well_gas[
                        (date_train <= df_well_gas.index) & (df_well_gas.index <= date_end)]
                    data_shelf[well] = {
                        LAST_MEASUREMENT: {
                            DATE: datetime_to_str(df_well_work_before_test.index[-1]),
                            VALUE: df_well_work_before_test[DEBIT].iloc[-1],
                            VALUE_LIQ: df_well_work_before_test[DEBIT1].iloc[-1],
                            VALUE_GAS: df_well_work_before_test_gas.iloc[-1] if not df_well_work_before_test_gas.empty else 0.0},
                        DEC_RATES: {
                            datetime_to_str(date_test): dec_rate},
                        DEC_RATES_LIQ: {
                            datetime_to_str(date_test): dec_rate_liq},
                        DEC_RATES_GAS: {
                            datetime_to_str(date_test): dec_rate_gas},
                        GTMS: dict(),
                    }
                    for stop_prd_start_date, stop_prd_end_date in self.get_stop_prd_dates(df_well):
                        data_shelf[well]['GTMS'][datetime_to_str(stop_prd_start_date)] = {
                            NAME: 'Текущий ремонт скважин',
                            'дебит нефти в период ТРС': 0,
                            'дебит жидкости в период ТРС': 0,
                            'длительность ТРС': (stop_prd_end_date - stop_prd_start_date).days + 1,
                            'длительность выхода на режим': 1
                        }
        transform_str_dates_to_datetime_or_vice_versa(data_shelf, dates_to_datetime=True)
        self._ensure_gas_defaults(data_shelf)
        self.df_fact_test_prd = df_fact_test_prd
        self.df_fact_test_prd_liq = df_fact_test_prd_liq
        self.df_fact_test_prd_gas = df_fact_test_prd_gas

        #если не было изменений в списке ГТМ, то используется для расчетов полученный словарь
        if st.session_state['change_gtm_info'] == 0:
            st.session_state.shelf_json = data_shelf
        else:
            self._ensure_gas_defaults(st.session_state.shelf_json)
            for well in data_shelf.keys():
                if well not in st.session_state.shelf_json.keys():
                    st.session_state.shelf_json[well] = data_shelf[well]
                    # print(f"{well} new")
            wells_to_delete = []
            for well in st.session_state.shelf_json.keys():
                if well not in data_shelf.keys():
                    wells_to_delete.append(well)
            for well_ in wells_to_delete:
                del st.session_state.shelf_json[well_]
                # print(f"DataProcessor: {well_} is deleted")
            # for well in data_shelf.keys():
            #     print(f"data_shelf: well = {well}; dec_rate = {data_shelf[well]}")
            # for well in st.session_state.shelf_json.keys():
            #     print(f"session.shelf_json: well = {well}; dec_rate = {st.session_state.shelf_json[well][DEC_RATES]}")




