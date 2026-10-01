import streamlit as st
import pandas as pd
import datetime as dt
import math
import numpy as np
from copy import deepcopy

from frameworks_shelf_algo.class_Shelf.constants import *
from frameworks_shelf_algo.class_Shelf.support_functions import get_date_range, get_s_decline_rates, \
    get_s_decline_rates_liq, get_s_decline_rates_gas
from frameworks_shelf_algo.class_Shelf.config import ConfigShelf
from frameworks_shelf_algo.class_Shelf.data_processor_shelf import DataProcessorShelf


class CalculatorShelf:
    def __init__(self, config: ConfigShelf):
        self._config = config
        self.wells_list = self._config.well_ois
        # print("Calculator Shelf")
        self._data_processor = DataProcessorShelf(self._config)
        self._get_input_data(self._data_processor)
        date_test = self._config.train_end
        date_end = self._config.predict_end
        self.df_result = pd.DataFrame()
        self.df_result_liq = pd.DataFrame()
        self.df_result_gas = pd.DataFrame()
        self.start_calcs(date_test, date_end)
        # для настроек ГТМ чтобы использовать функцию препроцессора
        st.session_state['change_gtm'] =  st.session_state['change_gtm'] + 1

    def _get_input_data(self, data_processor):
        # если внесены изменения в ГТМ, то учитываем их
        if st.session_state['change_gtm_info'] > 0:
            self._data_shelf = st.session_state.shelf_json
        else:
            self._data_shelf = data_processor.data_shelf
        self._df_fact_test_prd= data_processor.df_fact_test_prd
        self._df_fact_test_prd_liq = data_processor.df_fact_test_prd_liq
        self._df_fact_test_prd_gas = getattr(data_processor, 'df_fact_test_prd_gas', pd.DataFrame())

    def start_calcs(self, _date_start: dt.date, _date_end: dt.date):
        all_wells_data = self.get_all_wells_data(self._data_shelf)
        for well_name, well_data in all_wells_data.items():
            if well_name in self.wells_list:
                print(f"{well_name} is calculating")
                s_debit, s_debit_liq, s_debit_gas = self.get_s_debit(well_data, _date_end)
                try:
                    s_dec_rates = get_s_decline_rates(self._data_shelf, well_name,
                                                      s_debit.index[0], s_debit.index[-1])
                    s_dec_rates_liq = get_s_decline_rates_liq(self._data_shelf, well_name,
                                                              s_debit_liq.index[0], s_debit_liq.index[-1])
                    s_dec_rates_gas = get_s_decline_rates_gas(self._data_shelf, well_name,
                                                              s_debit_gas.index[0], s_debit_gas.index[-1])
                    self.fill_s_debit(s_debit, s_debit_liq, s_dec_rates, s_dec_rates_liq, well_data[GTMS])
                    self.fill_s_debit_gas(s_debit_gas, s_dec_rates_gas)
                    self.df_result[well_name] = s_debit.loc[s_debit.index >= _date_start]
                    self.df_result_liq[well_name] = s_debit_liq.loc[s_debit_liq.index >= _date_start]
                    self.df_result_gas[well_name] = s_debit_gas.loc[s_debit_gas.index >= _date_start]
                except IndexError:
                    continue
            else:
                print(f"{well_name} is NOT calculating")


    def get_all_wells_data(self, data) -> dict:
        all_wells_data = deepcopy(data)
        del all_wells_data[PLANNED_MLSP_STOPS]
        for well_name, well_data in all_wells_data.items():
            for mlsp_date, mlsp_data in st.session_state.shelf_json[PLANNED_MLSP_STOPS].items():
                well_data[GTMS][mlsp_date] = {
                    NAME: MLSP_STOP,
                    DATE_START_MLSP: mlsp_data[DATE_START_MLSP],
                }
                if well_name in mlsp_data:
                    well_data[GTMS][mlsp_date][DEBIT_INCREASE] = mlsp_data[well_name][DEBIT_INCREASE]
                    well_data[GTMS][mlsp_date][DEBIT_INCREASE_LIQ] = mlsp_data[well_name][DEBIT_INCREASE_LIQ]
                    well_data[GTMS][mlsp_date][DEBIT_INCREASE_GAS] = mlsp_data[well_name].get(DEBIT_INCREASE_GAS, 0.0)
                    well_data[GTMS][mlsp_date][N_DAYS_DEBIT_RECOVERY] = mlsp_data[well_name][N_DAYS_DEBIT_RECOVERY]
            well_data[GTMS] = {date: dict_ for date, dict_ in sorted(well_data[GTMS].items())}
        return all_wells_data


    def get_s_debit(self, well_data: dict, _date_end: dt.date):
        last_launch_date = None
        last_measurement_date = well_data[LAST_MEASUREMENT][DATE] if len(well_data[LAST_MEASUREMENT]) != 0 else None
        if last_launch_date is None and last_measurement_date is None:
            raise ValueError('Не найден ГТМ "Ввод новых скважин", нет данных по последнему известному значению дебита')
        else:
            _date_start_new = np.max([date for date in (last_measurement_date, last_launch_date) if date is not None])
            s_debit = pd.Series(index=get_date_range(_date_start_new, _date_end), dtype=float)
            s_debit_liq = pd.Series(index=get_date_range(_date_start_new, _date_end), dtype=float)
            s_debit_gas = pd.Series(index=get_date_range(_date_start_new, _date_end), dtype=float)
            if _date_start_new == last_measurement_date:
                s_debit[_date_start_new] = well_data[LAST_MEASUREMENT][VALUE]
                s_debit_liq[_date_start_new] = well_data[LAST_MEASUREMENT][VALUE_LIQ]
                s_debit_gas[_date_start_new] = well_data[LAST_MEASUREMENT].get(VALUE_GAS, 0.0)
            return s_debit, s_debit_liq, s_debit_gas

    @staticmethod
    def fill_s_debit_gas(s_debit_gas: pd.Series, s_dec_rates_gas: pd.Series) -> None:
        for pos_idx, date in enumerate(s_debit_gas.index):
            if not math.isnan(s_debit_gas[pos_idx]):
                continue
            n_considered_days = 1 if (date - s_debit_gas.index[0]).days <= 30 else 3
            s_debit_gas[pos_idx] = np.mean(s_debit_gas[pos_idx - n_considered_days:pos_idx]) - s_dec_rates_gas[date]

    def fill_s_debit(self, s_debit: pd.Series, s_debit_liq: pd.Series,
                     s_dec_rates: pd.Series, s_dec_rates_liq: pd.Series, gtms_data: dict) -> None:
        def get_n_considered_days(s_debit, date, mlsp_vnr_is_going):
            if MLSP_VNR_LAST_DATE in st.session_state and (date - st.session_state[MLSP_VNR_LAST_DATE]).days <= 3:
                mlsp_vnr_recently_finished = True
            else:
                mlsp_vnr_recently_finished = False
            if (date - s_debit.index[0]).days <= 30 or mlsp_vnr_is_going or mlsp_vnr_recently_finished:
                n_considered_days = 1
            else:
                n_considered_days = 3
            return n_considered_days

        def set_const_val_to_prediction(_date_start: dt.date, _date_end: dt.date, val_to_set: float,
                                        val_to_set_liq: float) -> None:
            s_debit.loc[(_date_start <= s_debit.index) & (s_debit.index <= _date_end)] = val_to_set
            s_debit_liq.loc[(_date_start <= s_debit_liq.index) & (s_debit_liq.index <= _date_end)] = val_to_set_liq

        def add_gtm_data_to_s_debit(s_debit: pd.Series, s_debit_liq: pd.Series,
                                    gtm_date: dt.date, gtm_data: dict) -> None:
            def add_vnr_to_s_debit(_debit_before: float,_debit_before_liq: float,
                                   _date_start_vnr: dt.date, _n_days_vnr: int) -> None:
                _date = None
                for n_days_past in range(1, _n_days_vnr + 1):
                    _date = _date_start_vnr + dt.timedelta(days=n_days_past - 1)
                    if _date in s_debit.index:
                        s_debit[_date] = _debit_before / _n_days_vnr * n_days_past
                        s_debit_liq[_date] = _debit_before_liq / _n_days_vnr * n_days_past
                if not st.session_state[MLSP_VNR_IS_GOING]:
                    set_const_val_to_prediction(
                        _date_start=_date + dt.timedelta(days=1),
                        _date_end=_date + dt.timedelta(days=2),
                        val_to_set=_debit_before,
                        val_to_set_liq=_debit_before_liq
                    )

            name = gtm_data[NAME]
            # if name == 'Ввод новых скважин':
            #     # ввод новых скважин не работает
            #     debit_after_vnr = gtm_data['дебит']
            #     n_days_vnr = gtm_data['длительность выхода на режим']
            #     add_vnr_to_s_debit(debit_after_vnr, gtm_date, n_days_vnr)
            # elif
            if name == 'Текущий ремонт скважин':
                debit_before = s_debit[gtm_date - dt.timedelta(days=1)]
                debit_before_liq = s_debit_liq[gtm_date - dt.timedelta(days=1)]
                debit_during_trs = gtm_data['дебит нефти в период ТРС']
                debit_during_trs_liq = gtm_data['дебит жидкости в период ТРС']
                n_days_trs = gtm_data['длительность ТРС']
                n_days_vnr = gtm_data['длительность выхода на режим']
                set_const_val_to_prediction(gtm_date, gtm_date + dt.timedelta(days=n_days_trs - 1),
                                            debit_during_trs, debit_during_trs_liq)
                add_vnr_to_s_debit(debit_before, debit_before_liq, gtm_date + dt.timedelta(days=n_days_trs), n_days_vnr)
                if st.session_state[MLSP_VNR_IS_GOING]:
                    set_const_val_to_prediction(gtm_date + dt.timedelta(days=n_days_trs + n_days_vnr),
                                                gtm_date + dt.timedelta(days=n_days_trs + n_days_vnr + 1),
                                                debit_before, debit_before_liq)
            elif name == 'Капитальный ремонт скважин':
                debit_before = s_debit[gtm_date - dt.timedelta(days=1)]
                debit_before_liq = s_debit_liq[gtm_date - dt.timedelta(days=1)]
                debit_during_krs = gtm_data['дебит нефти в период КРС']
                debit_during_krs_liq = gtm_data['дебит жидкости в период КРС']
                effect_krs = gtm_data['эффект по нефти от КРС']
                effect_krs_liq = gtm_data['эффект по жидкости от КРС']
                if st.session_state[MLSP_VNR_IS_GOING]:
                    st.session_state[TARGET_DEBIT_AFTER_MLSP_START] += effect_krs
                    st.session_state[TARGET_DEBIT_AFTER_MLSP_START_LIQ] += effect_krs_liq
                n_days_krs = gtm_data['длительность КРС']
                n_days_vnr = gtm_data['длительность выхода на режим']
                set_const_val_to_prediction(gtm_date, gtm_date + dt.timedelta(days=n_days_krs - 1),
                                            debit_during_krs, debit_during_krs_liq)
                add_vnr_to_s_debit(debit_before + effect_krs, debit_before_liq + effect_krs_liq,
                                   gtm_date + dt.timedelta(days=n_days_krs), n_days_vnr)
            elif name == 'Перевод в нагнетательный фонд':
                debit_during_and_after = gtm_data['дебит нефти в период и после работ']
                debit_during_and_after_liq = gtm_data['дебит жидкости в период и после работ']
                set_const_val_to_prediction(gtm_date, s_debit.index[-1], debit_during_and_after,
                                            debit_during_and_after_liq)
            elif name == 'Соляно-кислотная обработка':
                debit_before_sko = s_debit[gtm_date - dt.timedelta(days=1)]
                debit_before_sko_liq = s_debit_liq[gtm_date - dt.timedelta(days=1)]
                debit_during_sko = gtm_data['дебит нефти в период СКО']
                debit_during_sko_liq = gtm_data['дебит жидкости в период СКО']
                n_days_sko = gtm_data['длительность СКО']
                effect_sko = gtm_data['эффект по нефти от СКО']
                effect_sko_liq = gtm_data['эффект по жидкости от СКО']
                if st.session_state[MLSP_VNR_IS_GOING]:
                    st.session_state[TARGET_DEBIT_AFTER_MLSP_START] += effect_sko
                    st.session_state[TARGET_DEBIT_AFTER_MLSP_START_LIQ] += effect_sko_liq
                set_const_val_to_prediction(gtm_date, gtm_date + dt.timedelta(days=n_days_sko - 1),
                                            debit_during_sko, debit_during_sko_liq)
                date_start_and_end = gtm_date + dt.timedelta(days=n_days_sko)
                set_const_val_to_prediction(date_start_and_end, date_start_and_end, debit_before_sko,
                                            debit_before_sko_liq)
                date_start = gtm_date + dt.timedelta(days=n_days_sko + 1)
                date_end = gtm_date + dt.timedelta(days=n_days_sko + 3)
                set_const_val_to_prediction(date_start, date_end, debit_before_sko + effect_sko,
                                            debit_before_sko_liq + effect_sko_liq)
            elif name == 'Промыслово-геофизические исследования':
                debit_before_stop = s_debit.loc[gtm_date - dt.timedelta(days=1)]
                debit_before_stop_liq = s_debit_liq.loc[gtm_date - dt.timedelta(days=1)]
                debit_during_stop = gtm_data['дебит нефти во время остановки']
                debit_during_stop_liq = gtm_data['дебит жидкости во время остановки']
                n_days_stop = gtm_data['длительность остановки']
                set_const_val_to_prediction(gtm_date, gtm_date, debit_during_stop, debit_during_stop_liq)
                date_start_and_end = gtm_date + dt.timedelta(days=1)
                set_const_val_to_prediction(date_start_and_end, date_start_and_end, debit_before_stop / 2,
                                            debit_before_stop_liq / 2)
                date_start = gtm_date + dt.timedelta(days=2)
                date_end = gtm_date + dt.timedelta(days=n_days_stop)
                set_const_val_to_prediction(date_start, date_end, debit_during_stop, debit_during_stop_liq)
                date_start = gtm_date + dt.timedelta(days=n_days_stop + 1)
                date_end = gtm_date + dt.timedelta(days=n_days_stop + 2)
                set_const_val_to_prediction(date_start, date_end, debit_before_stop / 2, debit_before_stop_liq / 2)
                date_start = gtm_date + dt.timedelta(days=n_days_stop + 3)
                date_end = gtm_date + dt.timedelta(days=n_days_stop + 5)
                set_const_val_to_prediction(date_start, date_end, debit_before_stop, debit_before_stop_liq)
            elif name == MLSP_STOP:
                date_start_mlsp = gtm_data[DATE_START_MLSP]
                debit_before = s_debit.loc[gtm_date - dt.timedelta(days=1)]
                debit_before_liq = s_debit_liq.loc[gtm_date - dt.timedelta(days=1)]
                set_const_val_to_prediction(gtm_date, date_start_mlsp - dt.timedelta(days=1), 0, 0)
                if (DEBIT_INCREASE not in gtm_data) and (DEBIT_INCREASE_LIQ not in gtm_data):
                    set_const_val_to_prediction(date_start_mlsp, date_start_mlsp + dt.timedelta(days=2),
                                                debit_before, debit_before_liq)
                else:
                    set_const_val_to_prediction(date_start_mlsp, date_start_mlsp, debit_before, debit_before_liq)
                    debit_increase = gtm_data[DEBIT_INCREASE]
                    debit_increase_liq = gtm_data[DEBIT_INCREASE_LIQ]
                    date_big_debit = date_start_mlsp + dt.timedelta(days=1)
                    set_const_val_to_prediction(date_big_debit, date_big_debit, debit_before + debit_increase,
                                                debit_before_liq + debit_increase_liq)
                    st.session_state[MLSP_VNR_IS_GOING] = True
                    st.session_state[MLSP_VNR_FIRST_DATE] = date_big_debit + dt.timedelta(days=1)
                    st.session_state[DEC_RATE_DURING_MLSP_VNR] = debit_increase / (gtm_data[N_DAYS_DEBIT_RECOVERY] - 1)
                    st.session_state[DEC_RATE_DURING_MLSP_VNR_LIQ] = \
                        debit_increase_liq / (gtm_data[N_DAYS_DEBIT_RECOVERY] - 1)
                    st.session_state[TARGET_DEBIT_AFTER_MLSP_START] = debit_before
                    st.session_state[TARGET_DEBIT_AFTER_MLSP_START_LIQ] = debit_before_liq

        def transfer_mlsp_vnrs_data_to_initial_state() -> None:
            st.session_state[MLSP_VNR_IS_GOING] = False
            if MLSP_VNR_FIRST_DATE in st.session_state:
                del st.session_state[MLSP_VNR_FIRST_DATE]
            if MLSP_VNR_LAST_DATE in st.session_state:
                del st.session_state[MLSP_VNR_LAST_DATE]
            if DEC_RATE_DURING_MLSP_VNR in st.session_state:
                del st.session_state[DEC_RATE_DURING_MLSP_VNR]
            if DEC_RATE_DURING_MLSP_VNR_LIQ in st.session_state:
                del st.session_state[DEC_RATE_DURING_MLSP_VNR_LIQ]
            if TARGET_DEBIT_AFTER_MLSP_START in st.session_state:
                del st.session_state[TARGET_DEBIT_AFTER_MLSP_START]
            if TARGET_DEBIT_AFTER_MLSP_START_LIQ in st.session_state:
                del st.session_state[TARGET_DEBIT_AFTER_MLSP_START_LIQ]

        transfer_mlsp_vnrs_data_to_initial_state()
        for pos_idx, date in enumerate(s_debit.index):
            if not math.isnan(s_debit[pos_idx]):
                continue
            else:
                mlsp_vnr_is_going = st.session_state[MLSP_VNR_IS_GOING]
                n_considered_days = get_n_considered_days(s_debit, date, mlsp_vnr_is_going)
                if date in gtms_data:
                    add_gtm_data_to_s_debit(s_debit, s_debit_liq, date, gtms_data[date])
                else:
                    dec_rate = st.session_state[DEC_RATE_DURING_MLSP_VNR] if mlsp_vnr_is_going else s_dec_rates[date]
                    dec_rate_liq = st.session_state[DEC_RATE_DURING_MLSP_VNR_LIQ] \
                        if mlsp_vnr_is_going else s_dec_rates_liq[date]
                    s_debit[pos_idx] = np.mean(s_debit[pos_idx - n_considered_days:pos_idx]) - dec_rate
                    s_debit_liq[pos_idx] = np.mean(s_debit_liq[pos_idx - n_considered_days:pos_idx]) - dec_rate_liq
                    if mlsp_vnr_is_going:
                        debits_are_equal = abs(
                            s_debit[pos_idx] - st.session_state[TARGET_DEBIT_AFTER_MLSP_START]) < 10e-6
                        if s_debit[pos_idx] < st.session_state[TARGET_DEBIT_AFTER_MLSP_START] or debits_are_equal:
                            st.session_state[MLSP_VNR_IS_GOING] = False
                            st.session_state[MLSP_VNR_LAST_DATE] = date - dt.timedelta(days=1)
                    if MLSP_VNR_LAST_DATE in st.session_state and (
                            date - st.session_state[MLSP_VNR_LAST_DATE]).days == 3:
                        set_const_val_to_prediction(date - dt.timedelta(days=2), date, s_debit[pos_idx - 2],
                                                    s_debit_liq[pos_idx - 2])

                    # dec_rate = s_dec_rates[date]
                    # dec_rate_liq = s_dec_rates_liq[date]
                    # s_debit[pos_idx] = np.mean(s_debit[pos_idx - n_considered_days:pos_idx]) - dec_rate
                    # s_debit_liq[pos_idx] = np.mean(s_debit_liq[pos_idx - n_considered_days:pos_idx]) - dec_rate_liq

