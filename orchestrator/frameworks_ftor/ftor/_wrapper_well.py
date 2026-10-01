import datetime
import numpy as np
import pandas as pd
import json

from models_waterflood.waterflood.corey import CoreyModel
from frameworks_ftor.ftor.config import Config
from frameworks_ftor.ftor.well import Well
from frameworks_ftor.ftor.well import WellResults
from frameworks_ftor.ftor.wrapper_estimator import compute_transient_flow, PARAMS_DEFAULT, KIND_CODE, K, L_HOR, \
    R_RES, PI, H, PHI, CT, MU, B, XF
from models_transflow_siam.transflow.no_jit.optimize_lestsqr import solve_lm_lu_with_bounds
from models_transflow_siam.transflow import constants as const


class WrapperWell(object):

    def __init__(
            self,
            config: Config,
            well: Well,
    ):
        self._config = config
        self._well = well
        self._run()

    def _run(self) -> None:
        self._compute_rate_liq()
        self._compute_rate_oil()
        self._add_results()

    def _compute_rate_liq(self) -> None:
        wrapper_well_rate_liq = _WrapperWellRateLiq(self._config, self._well)
        self._all_params = wrapper_well_rate_liq.all_params
        self._adap_and_fixed_params = wrapper_well_rate_liq.adap_and_fixed_params
        self._rates_liq_train = wrapper_well_rate_liq.rates_train
        self._rates_liq_test = wrapper_well_rate_liq.rates_test

    def _compute_rate_oil(self) -> None:
        corey_model = CoreyModel()
        cums_oil = self._well.df_flood[Well.NAME_CUM_OIL]
        watercuts = self._well.df_flood[Well.NAME_WATERCUT]
        corey_model.fit(
            cums_oil.values,
            watercuts.values,
        )
        self._params_model_corey = corey_model.get_params()
        self._watercuts_train = pd.Series(
            corey_model.compute(cums_oil.values),
            index=cums_oil.index,
        )
        # Метод CoreyModel.predict принимает массив последовательных возрастающих значений cums_liq по скважине.
        # На практике накопленная добыча может как возрастать, так и не меняться во времени, если скважина не работает.
        # Для решения такого противоречия в расчетах вводится допущение, что скважина не может иметь
        # неположительный дебит.
        rates_liq_test = self._rates_liq_test.replace(to_replace=0, value=0.001)
        # TODO: delete this filter and fix the end of the comment above
        rates_liq_test[rates_liq_test <= 0] = 0.001
        cums_liq = np.cumsum(rates_liq_test) + self._well.cum_liq_test
        self._watercuts_test = pd.Series(
            corey_model.predict(cums_liq, cum_oil_start=self._well.cum_oil_test),
            index=rates_liq_test.index,
        )
        self._rates_oil_test = self._rates_liq_test * self._watercuts_test.apply(
            lambda x: 1 - x) * self._well.density_oil

    def _add_results(self) -> None:
        well_results = WellResults(
            self._all_params,
            self._adap_and_fixed_params,
            self._params_model_corey,
            self._rates_liq_train,
            self._rates_liq_test,
            self._watercuts_train,
            self._watercuts_test,
            self._rates_oil_test,
        )
        self._well.results = well_results

    @property
    def well(self) -> Well:
        return self._well


class _WrapperWellRateLiq:
    _NAME_STATUS = Well.NAME_STATUS
    _NAME_PRESSURE = Well.NAME_PRESSURE
    _NAME_RATE_LIQ = Well.NAME_RATE_LIQ
    _NAME_CONSTRAINTS = Well.NAME_CONSTRAINTS
    _NAME_REL_TIME = 'Относительное время'
    _NAME_RATE_LIQ_MODEL = 'Дебит жидкости модельный'
    _NAME_DEPRESS_FROM_PRD = 'Депрессия от периода'
    _NAME_DEPRESS_FROM_PRD_OF_EQUAL_TIME = 'Депрессия от периода эквивалентного времени'

    _EQUAL_TIME0 = 365
    _N_STOP_DAYS = 10
    _K_EQUAL_TIME = 100

    _optimizer_params = {
        'bound_strategy': const.BOUND_STRATEGY_CYCLE_MIRROR,
        'max_iter': 25,
        'max_error': 0.3,
        'delta_error': 1e-6,
        'broyden_period': 5,
        'hessian_update_strategy': const.HESSIAN_UPDATE_STRATEGY_MARQUARDT,
    }

    def __init__(
            self,
            config: Config,
            well: Well,
    ):
        self._prepare_data(config, well)
        self._loop_over_periods_where_some_params_change()

    def _prepare_data(self, config: Config, well: Well) -> None:
        self._apply_last_points_adaptation = config.apply_last_points_adaptation
        self._number_of_last_points = config.number_of_last_points
        self._adap_bounds_data_last_points = {
            config.param_name_last_points_adaptation: config.param_bounds_last_points_adaptation
        }
        self._to_calc_equal_time, self._equal_p, self._cum_liq_start = self._prepare_data_for_eq_time_algo(config, well)
        self._date_start = well.date_start
        self._date_test = well.date_test
        self._df_chess = self._get_df_chess(well)
        if not self._to_calc_equal_time:
            self._set_relative_time_in_chess(equal_time=0)
        s_constrs = self._df_chess[self._NAME_CONSTRAINTS].dropna()
        self._adap_bounds_data_lst, self._params_to_fix_lst = self._unpack_constraints(s_constrs)
        self._big_prd_dates = self._get_big_prd_dates(well, s_constrs)
        self._work_prd_dates = self._get_work_prd_dates()
        self._n_big_prds, self._n_work_prds = len(self._big_prd_dates), len(self._work_prd_dates)
        self._qs_fact_big_prd = self._get_qs_fact_big_prd()
        self._last_big_prd_index_on_adaptation = self._get_last_prd_index_on_adaptation(self._big_prd_dates)
        self._last_work_prd_index_on_adaptation = self._get_last_prd_index_on_adaptation(self._work_prd_dates)
        self._work_prd_indexes_inside_big_prd_lst = self._get_work_prd_indexes_inside_big_prd_lst()
        self._set_depression_columns_in_df_chess()
        self._big_prd_params_lst = self._get_big_prd_params_lst(well)

    def _loop_over_periods_where_some_params_change(self) -> None:
        self._num_calls_loss = 0
        self._now_adap_on_last_points = False
        self._big_prd_idx = 0
        while self._big_prd_idx < self._n_big_prds:
            is_last_big_prd = self._big_prd_idx + 1 == self._n_big_prds
            yes_if_not_is_last_big_prd = False if is_last_big_prd else True
            self._big_prd_params_lst[self._big_prd_idx].update(self._params_to_fix_lst[self._big_prd_idx])
            adap_bounds_data = self._adap_bounds_data_lst[self._big_prd_idx]

            if len(adap_bounds_data) == 0:
                self._now_optimization = False
                self._to_calc_depression_for_next_big_prds = yes_if_not_is_last_big_prd
                self._get_loss_lst([])
            else:
                self._now_optimization = True
                self._to_calc_depression_for_next_big_prds = False
                bounds = np.array([adap_bounds_data[prm] for prm in adap_bounds_data])
                vals_init, bounds_corrected = self._get_init_point_and_correct_bounds(bounds)
                r = solve_lm_lu_with_bounds(
                    self._get_loss_lst,
                    vals_init,
                    bounds_corrected,
                    self._get_validated_bounds,
                    self._optimizer_params
                )
                vals_optimized = r[2]
                self._now_optimization = False
                self._to_calc_depression_for_next_big_prds = yes_if_not_is_last_big_prd
                self._get_loss_lst(vals_optimized)

            adap_on_last_points = False
            if self._apply_last_points_adaptation and self._big_prd_idx == self._last_big_prd_index_on_adaptation:
                last_big_prd_dates = self._qs_fact_big_prd[self._big_prd_idx].index
                last_points = last_big_prd_dates[last_big_prd_dates < self._date_test][-self._number_of_last_points:]
                adap_on_last_points = any(self._df_chess[self._NAME_RATE_LIQ_MODEL][last_points] > 0)
            if adap_on_last_points:
                self._now_optimization = True
                self._now_adap_on_last_points = True
                self._to_calc_depression_for_next_big_prds = False
                prm_name = list(self._adap_bounds_data_last_points.keys())[0]
                bounds = self._adap_bounds_data_last_points[prm_name]
                val_init = self._big_prd_params_lst[self._big_prd_idx][prm_name]
                print(f'\n----------- ADAPTATION ON LAST POINT BY {prm_name} ----------')
                r = solve_lm_lu_with_bounds(
                    self._get_loss_lst_pi_adap if prm_name == PI else self._get_loss_lst,
                    np.array([val_init]),
                    np.array([bounds]),
                    self._get_validated_bounds,
                    self._optimizer_params
                )
                val_optimized = r[2]
                self._now_optimization = False
                self._to_calc_depression_for_next_big_prds = yes_if_not_is_last_big_prd
                self._get_loss_lst(val_optimized)
                self._now_adap_on_last_points = False

            if not is_last_big_prd:
                self._big_prd_params_lst[self._big_prd_idx + 1] = self._big_prd_params_lst[self._big_prd_idx].copy()

            self._big_prd_idx += 1

    def _prepare_data_for_eq_time_algo(self, config: Config, well: Well) -> tuple[bool, float, float]:
        cum_liq_start = well.cum_liq_start
        if config.use_equal_time_algorithm and cum_liq_start > 0:
            to_calc_equal_time = True
            equal_p = well.df_chess.iloc[0][self._NAME_PRESSURE]
        else:
            to_calc_equal_time = False
            equal_p = 0
        return to_calc_equal_time, equal_p, cum_liq_start

    def _get_df_chess(self, well: Well) -> pd.DataFrame:
        df_chess = well.df_chess[[
            self._NAME_STATUS,
            self._NAME_PRESSURE,
            self._NAME_RATE_LIQ,
            self._NAME_CONSTRAINTS,
        ]].copy()
        df_chess[[
            self._NAME_REL_TIME,
            self._NAME_RATE_LIQ_MODEL,
        ]] = None
        return df_chess

    def _set_relative_time_in_chess(self, equal_time: int) -> None:
        if equal_time == 0:
            date_start_rel_time = self._date_start
        else:
            date_start_rel_time = self._date_start - datetime.timedelta(days=equal_time + self._N_STOP_DAYS)
        rel_t_timedelta = self._df_chess.index - (date_start_rel_time - datetime.timedelta(days=1))
        rel_t_hours = [*map(lambda dt: dt.days * 24, rel_t_timedelta)]
        self._df_chess[self._NAME_REL_TIME] = rel_t_hours

    @staticmethod
    def _unpack_constraints(s_constrs: pd.Series) -> tuple[
        list[dict[str, list[float]]],
        list[dict[str, float]]
    ]:
        adap_bounds_data_lst = []
        params_to_fix_lst = []
        for constrs_string in s_constrs.tolist():
            constrs = json.loads(constrs_string)
            ordered_constrs = dict()
            for prm_name in sorted(constrs.keys(), key=lambda prm_name: list(PARAMS_DEFAULT.keys()).index(prm_name)):
                ordered_constrs[prm_name] = constrs[prm_name]
            constrs = ordered_constrs
            adap_bounds_data = dict()
            params_to_fix = dict()
            for prm in constrs:
                if isinstance(constrs[prm], list):
                    adap_bounds_data[prm] = constrs[prm]
                else:
                    params_to_fix[prm] = constrs[prm]
            adap_bounds_data_lst.append(adap_bounds_data)
            params_to_fix_lst.append(params_to_fix)
        return adap_bounds_data_lst, params_to_fix_lst

    @staticmethod
    def _get_big_prd_dates(well: Well, s_constrs: pd.Series) -> list[tuple[datetime.date, datetime.date]]:
        big_prd_dates = []
        boarder_dates = s_constrs.index.tolist() + [well.date_end + datetime.timedelta(days=1)]
        n_big_prds = len(boarder_dates) - 1
        for i in range(n_big_prds):
            start_date_big_prd = boarder_dates[i]
            end_date_big_prd = boarder_dates[i + 1] - datetime.timedelta(days=1)
            big_prd_dates.append((start_date_big_prd, end_date_big_prd))
        return big_prd_dates

    def _get_work_prd_dates(self) -> list[tuple[datetime.date, datetime.date]]:
        col_date = 'date'
        df_statuses = self._df_chess[[self._NAME_STATUS]].copy()
        df_statuses[col_date] = self._df_chess.index
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
                elif df_statuses.loc[i, self._NAME_STATUS] == 'В работе':
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
                    elif df_statuses.loc[i, self._NAME_STATUS] == 'Остановлена':
                        end_date = df_statuses.loc[i - 1, col_date]
                        work_dates.append((start_date, end_date))
                        i += 1
                        break
                    else:
                        i += 1
        return work_dates

    def _get_qs_fact_big_prd(self) -> list[pd.Series]:
        qs_fact_big_prd = []
        for big_prd_start_date, big_prd_end_date in self._big_prd_dates:
            df_chess_big_prd = self._df_chess.loc[(big_prd_start_date <= self._df_chess.index) &
                                                  (self._df_chess.index <= big_prd_end_date)]
            qs_fact_big_prd.append(df_chess_big_prd[self._NAME_RATE_LIQ])
        return qs_fact_big_prd

    def _get_last_prd_index_on_adaptation(self, dates: list[tuple[datetime.date, datetime.date]]) -> int:
        last_prd_index_on_adaptation = None
        for prd_idx, (start_date, end_date) in enumerate(dates):
            if start_date < self._date_test:
                last_prd_index_on_adaptation = prd_idx
        return last_prd_index_on_adaptation

    def _get_work_prd_indexes_inside_big_prd_lst(self) -> list[list[int]]:
        work_prd_indexes_inside_big_prd_lst = []
        work_prd_idx = 0
        for big_prd_idx, (big_prd_start_date, big_prd_end_date) in enumerate(self._big_prd_dates):
            work_prd_indexes_inside_big_prd_lst.append([])
            while work_prd_idx < self._n_work_prds:
                work_prd_start_date = self._work_prd_dates[work_prd_idx][0]
                work_prd_is_inside_big_prd = big_prd_start_date <= work_prd_start_date <= big_prd_end_date
                if work_prd_is_inside_big_prd:
                    work_prd_indexes_inside_big_prd_lst[big_prd_idx].append(work_prd_idx)
                    work_prd_idx += 1
                else:
                    break
        return work_prd_indexes_inside_big_prd_lst

    def _set_depression_columns_in_df_chess(self) -> None:
        self._df_chess[self._NAME_DEPRESS_FROM_PRD_OF_EQUAL_TIME] = 0.
        for work_prd_idx in range(self._n_work_prds - 1):
            self._df_chess[f'{self._NAME_DEPRESS_FROM_PRD} {work_prd_idx}'] = np.nan

    def _get_big_prd_params_lst(self, well: Well) -> list[dict[str, float] or None]:
        changed_default_params = {
            H: well.thickness,
            PHI: well.porosity,
            CT: well.compressibility_total,
            MU: well.viscosity_liq,
            B: well.volume_factor_liq
        }
        first_big_prd_params_before_optimization = PARAMS_DEFAULT.copy()
        first_big_prd_params_before_optimization.update(changed_default_params)
        return [first_big_prd_params_before_optimization] + [None] * (self._n_big_prds - 1)

    def _get_loss_lst(self, adap_param_vals: list[float]) -> float:
        self._num_calls_loss += 1
        ad_prms = self._get_adap_params_dict(adap_param_vals)
        self._big_prd_params_lst[self._big_prd_idx].update(ad_prms)
        all_params = self._big_prd_params_lst[self._big_prd_idx]
        q_fact_big_prd = self._qs_fact_big_prd[self._big_prd_idx]
        q_model_big_prd = pd.Series(index=q_fact_big_prd.index)

        ad_prms_rounded = {ad_prm_name: round(ad_prms[ad_prm_name], 1) for ad_prm_name in ad_prms}
        print(f'prd {self._big_prd_idx + 1} of {self._n_big_prds}, iter {self._num_calls_loss}: {ad_prms_rounded}')

        work_prd_idxs_inside_big_prd = self._work_prd_indexes_inside_big_prd_lst[self._big_prd_idx]
        for work_prd_idx in work_prd_idxs_inside_big_prd:
            to_cut_redundant_calcs = work_prd_idx == self._last_work_prd_index_on_adaptation and self._now_optimization
            work_prd_start_date = self._work_prd_dates[work_prd_idx][0]
            if to_cut_redundant_calcs:
                work_prd_end_date = self._date_test - datetime.timedelta(days=1)
            else:
                work_prd_end_date = self._work_prd_dates[work_prd_idx][1]
            is_first_work_prd = work_prd_idx == 0
            is_last_work_prd_inside_big_prd = work_prd_idx == work_prd_idxs_inside_big_prd[-1]

            if is_first_work_prd and self._to_calc_equal_time:
                time_before0 = np.array([24 * (n + 1) for n in range(self._EQUAL_TIME0)])
                p_history_before0 = [(self._EQUAL_TIME0, self._equal_p)]
                prms_before = all_params.copy()
                prms_before.update({K: self._K_EQUAL_TIME})
                q_vals0 = compute_transient_flow(time_before0, p_history_before0, is_rate=True, **prms_before)
                equal_t = int(self._cum_liq_start / np.sum(q_vals0) * self._EQUAL_TIME0)
                if equal_t > 0:
                    self._set_relative_time_in_chess(equal_t)
                    equal_q = self._cum_liq_start / equal_t
                    time_calc_p = self._df_chess[self._NAME_REL_TIME].to_numpy()
                    n_days_zero_q = self._N_STOP_DAYS + len(time_calc_p)
                    q_history = [(equal_t, equal_q), (n_days_zero_q, 0)]
                    p_vals = compute_transient_flow(time_calc_p, q_history, is_rate=False, **prms_before)
                    depression_from_eq_t_prd = all_params[PI] - p_vals
                    self._df_chess[self._NAME_DEPRESS_FROM_PRD_OF_EQUAL_TIME] = depression_from_eq_t_prd
                else:
                    self._set_relative_time_in_chess(equal_time=0)

            df_chess_work_prd = self._get_slice_of_df_chess(work_prd_start_date, work_prd_end_date)
            cum_residual_depress = df_chess_work_prd[self._NAME_DEPRESS_FROM_PRD_OF_EQUAL_TIME].to_numpy()
            for previous_work_prd_idx in range(work_prd_idx):
                col_name = f'{self._NAME_DEPRESS_FROM_PRD} {previous_work_prd_idx}'
                residual_depress = df_chess_work_prd[col_name].to_numpy()
                cum_residual_depress += residual_depress
            p_history = []
            p_vals = df_chess_work_prd[self._NAME_PRESSURE].to_numpy()
            well_birth_rel_time = df_chess_work_prd.iloc[0][self._NAME_REL_TIME] - 24
            for p_val_corrected in p_vals + cum_residual_depress:
                p_history.append((1, p_val_corrected))

            time_calc_q = df_chess_work_prd[self._NAME_REL_TIME].to_numpy() - well_birth_rel_time
            q_vals = compute_transient_flow(time_calc_q, p_history, is_rate=True, **all_params)
            q_model_big_prd.loc[df_chess_work_prd.index] = q_vals

            if to_cut_redundant_calcs:
                break

            if self._to_calc_depression_for_next_big_prds:
                to_calc_depress = True
                last_work_prd_idx_calc_depress = self._work_prd_indexes_inside_big_prd_lst[-1][-1]
            elif not is_last_work_prd_inside_big_prd:
                to_calc_depress = True
                last_work_prd_idx_calc_depress = work_prd_idxs_inside_big_prd[-1]
            else:
                to_calc_depress = False
                last_work_prd_idx_calc_depress = None

            if to_calc_depress:
                start_date_calc_depress = self._work_prd_dates[work_prd_idx + 1][0]
                end_date_calc_depress = self._work_prd_dates[last_work_prd_idx_calc_depress][1]
                end_date_work_prd = self._work_prd_dates[work_prd_idx][1]
                n_days_zero_q = (end_date_calc_depress - end_date_work_prd).days
                q_history = []
                for q_val in q_vals:
                    q_history.append((1, q_val))
                q_history.append((n_days_zero_q, 0))
                df_chess_calc_depress = self._get_slice_of_df_chess(start_date_calc_depress, end_date_calc_depress)
                time_calc_p = df_chess_calc_depress[self._NAME_REL_TIME].to_numpy() - well_birth_rel_time
                p_vals = compute_transient_flow(time_calc_p, q_history, is_rate=False, **all_params)
                dates = pd.date_range(start_date_calc_depress, end_date_calc_depress, freq="D")
                self._df_chess[f'{self._NAME_DEPRESS_FROM_PRD} {work_prd_idx}'][dates] = all_params[PI] - p_vals

        self._df_chess[self._NAME_RATE_LIQ_MODEL].loc[q_model_big_prd.index] = q_model_big_prd.fillna(0)
        if self._big_prd_idx != self._last_big_prd_index_on_adaptation:
            q_model_to_calc_loss = q_model_big_prd
        else:
            q_model_to_calc_loss = q_model_big_prd.loc[q_model_big_prd.index < self._date_test]
            if self._now_adap_on_last_points:
                q_model_to_calc_loss = q_model_to_calc_loss[-self._number_of_last_points:]
        loss_lst = (q_model_to_calc_loss - q_fact_big_prd).dropna().tolist()
        return loss_lst

    def _get_adap_params_dict(self, adap_param_vals: list[float]) -> dict[str, float]:
        if self._now_adap_on_last_points:
            adap_bounds_data = self._adap_bounds_data_last_points
        else:
            adap_bounds_data = self._adap_bounds_data_lst[self._big_prd_idx]
        adap_params = {prm_name: prm_val for prm_name, prm_val in zip(adap_bounds_data, adap_param_vals)}
        return adap_params

    def _get_slice_of_df_chess(self, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        slice_of_df_chess = self._df_chess.loc[(start_date <= self._df_chess.index) &
                                               (self._df_chess.index <= end_date)]
        return slice_of_df_chess

    def _get_init_point_and_correct_bounds(self, bounds_lst_input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._now_adap_on_last_points:
            adap_bounds_data = self._adap_bounds_data_last_points
        else:
            adap_bounds_data = self._adap_bounds_data_lst[self._big_prd_idx]
        ad_prm_names = list(adap_bounds_data.keys())
        all_prms = self._big_prd_params_lst[self._big_prd_idx]
        bounds_lst = np.ndarray.copy(bounds_lst_input)

        r_res_max = bounds_lst[ad_prm_names.index(R_RES)][1] if R_RES in ad_prm_names else all_prms[R_RES]
        kind_code = all_prms[KIND_CODE]

        if kind_code in [1, 3]:
            if R_RES in ad_prm_names:
                l_hor_min = bounds_lst[ad_prm_names.index(L_HOR)][0] if L_HOR in ad_prm_names else all_prms[L_HOR]
                bounds = bounds_lst[ad_prm_names.index(R_RES)]
                if bounds[0] < l_hor_min:
                    bounds[0] = l_hor_min
            if L_HOR in ad_prm_names:
                bounds_l_hor = bounds_lst[ad_prm_names.index(L_HOR)]
                if bounds_l_hor[1] > r_res_max:
                    bounds_l_hor[1] = r_res_max

        if kind_code in [2, 3]:
            if R_RES in ad_prm_names:
                xf_min = bounds_lst[ad_prm_names.index(XF)][0] if XF in ad_prm_names else all_prms[XF]
                bounds = bounds_lst[ad_prm_names.index(R_RES)]
                if bounds[0] < 2 * xf_min:
                    bounds[0] = 2 * xf_min
            if XF in ad_prm_names:
                bounds_xf = bounds_lst[ad_prm_names.index(XF)]
                if bounds_xf[1] > r_res_max / 2:
                    bounds_xf[1] = r_res_max / 2
        self._check_if_bounds_are_correct(ad_prm_names, bounds_lst)

        tr_vals_init = np.array([np.mean(bounds) for bounds in bounds_lst])
        r_res_init = tr_vals_init[ad_prm_names.index(R_RES)] if R_RES in ad_prm_names else all_prms[R_RES]

        for idx, init_val in enumerate(tr_vals_init):
            ad_prm_name = ad_prm_names[idx]
            if ad_prm_name in [L_HOR, XF]:
                if init_val == r_res_init:
                    bounds = bounds_lst[idx]
                    tr_vals_init[idx] = bounds[0] + (bounds[1] - bounds[0]) * 0.4
        return tr_vals_init, bounds_lst

    @staticmethod
    def _check_if_bounds_are_correct(ad_prm_names: list[str], bounds_lst: list[list[float]]) -> None:
        for prm, bounds in zip(ad_prm_names, bounds_lst):
            if bounds[0] >= bounds[1]:
                raise ValueError(
                    f'left bound for {prm} is greater or equal to right one after transformation: {bounds = }'
                )

    def _get_validated_bounds(self, vals, bounds_lst_init, *args):
        adap_params = self._get_adap_params_dict(vals)
        ad_prm_names = list(adap_params.keys())
        all_prms = self._big_prd_params_lst[self._big_prd_idx]
        all_prms.update(adap_params)
        bounds_lst = np.ndarray.copy(bounds_lst_init)

        r_res = all_prms[R_RES]
        kind_code = all_prms[KIND_CODE]

        if kind_code in [1, 3]:
            if R_RES in ad_prm_names:
                l_hor = all_prms[L_HOR]
                bounds = bounds_lst[ad_prm_names.index(R_RES)]
                if bounds[0] < l_hor:
                    bounds[0] = l_hor
            if L_HOR in ad_prm_names:
                bounds_l_hor = bounds_lst[ad_prm_names.index(L_HOR)]
                if bounds_l_hor[1] > r_res:
                    bounds_l_hor[1] = r_res

        if kind_code in [2, 3]:
            if R_RES in ad_prm_names:
                xf = all_prms[XF]
                bounds = bounds_lst[ad_prm_names.index(R_RES)]
                if bounds[0] < 2 * xf:
                    bounds[0] = 2 * xf
            if XF in ad_prm_names:
                bounds_xf = bounds_lst[ad_prm_names.index(XF)]
                if bounds_xf[1] > r_res / 2:
                    bounds_xf[1] = r_res / 2
        return bounds_lst

    def _get_loss_lst_pi_adap(self, pi_lst: list):
        pi = pi_lst[0]
        self._to_calc_depression_for_next_big_prds = True
        self._now_adap_on_last_points = False
        self._big_prd_idx = 0
        while self._big_prd_idx < self._last_big_prd_index_on_adaptation:
            self._big_prd_params_lst[self._big_prd_idx].update({PI: pi})
            ad_prm_names = list(self._adap_bounds_data_lst[self._big_prd_idx].keys())
            ad_prm_vals = []
            for prm_name in ad_prm_names:
                ad_prm_val = self._big_prd_params_lst[self._big_prd_idx][prm_name]
                ad_prm_vals.append(ad_prm_val)
            self._get_loss_lst(ad_prm_vals)
            self._big_prd_idx += 1
        self._to_calc_depression_for_next_big_prds = False if self._big_prd_idx + 1 == self._n_big_prds else True
        self._now_adap_on_last_points = True
        return self._get_loss_lst([pi])

    @property
    def all_params(self) -> list[dict[str, float]]:
        return self._big_prd_params_lst

    @property
    def adap_and_fixed_params(self) -> list[dict[str, float]]:
        adap_and_fixed_params = []
        for big_prd_idx in range(len(self._big_prd_params_lst)):
            ad_prm_names = list(self._adap_bounds_data_lst[big_prd_idx].keys())
            if self._apply_last_points_adaptation and big_prd_idx == self._last_big_prd_index_on_adaptation:
                prm_name_for_last_points_adaptation = list(self._adap_bounds_data_last_points.keys())[0]
                ad_prm_names.append(prm_name_for_last_points_adaptation)
            fixed_prm_names = list(self._params_to_fix_lst[big_prd_idx].keys())
            ad_and_fixed_prm_names = ad_prm_names + fixed_prm_names
            ad_and_fixed_prm_names.sort(key=lambda prm_name: list(PARAMS_DEFAULT.keys()).index(prm_name))
            ad_and_fixed_prms = dict()
            for prm_name in ad_and_fixed_prm_names:
                ad_and_fixed_prms[prm_name] = self._big_prd_params_lst[big_prd_idx][prm_name]
            adap_and_fixed_params.append(ad_and_fixed_prms)
        return adap_and_fixed_params

    @property
    def rates_train(self) -> pd.Series:
        return self._df_chess[self._NAME_RATE_LIQ_MODEL].loc[self._df_chess.index < self._date_test]

    @property
    def rates_test(self) -> pd.Series:
        return self._df_chess[self._NAME_RATE_LIQ_MODEL].loc[self._df_chess.index >= self._date_test]
