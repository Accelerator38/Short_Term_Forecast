from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, List, Tuple, Dict

import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from loguru import logger

cache_singleton = getattr(st, "experimental_singleton", None)
if cache_singleton is None:
    cache_singleton = st.cache_resource

cache_memo = getattr(st, "experimental_memo", None)
if cache_memo is None:
    cache_memo = st.cache_data

if TYPE_CHECKING:
    from frameworks_crm.class_CRM.calculator import Calculator as CalculatorCRM
    from frameworks_ftor.ftor.calculator import Calculator as CalculatorFtor
    from frameworks_hybrid_crm_ml.class_Fedot.calculator import CalculatorFedot
    from frameworks_shelf_algo.class_Shelf.data_postprocessor_shelf import DataPostProcessorShelf
    from frameworks_wolfram.wolfram.calculator import Calculator as CalculatorWolfram
    from tools_preprocessor.config import Config as ConfigPreprocessor
    from tools_preprocessor.preprocessor import Preprocessor


def _preprocessor_cache_key(config: ConfigPreprocessor) -> tuple:
    return (
        config.field_name,
        tuple(config.shops),
        config.date_start,
        config.date_test,
        config.date_end,
    )


@cache_singleton(show_spinner=False)
def _run_preprocessor_cached(
        field_name: str,
        shops: tuple[str, ...],
        date_start: date,
        date_test: date,
        date_end: date,
) -> Preprocessor:
    from tools_preprocessor.preprocessor import Preprocessor
    from tools_preprocessor.config import Config as ConfigPreprocessor

    config = ConfigPreprocessor(field_name, list(shops), date_start, date_test, date_end)
    preprocessor = Preprocessor(config)
    return preprocessor


def run_preprocessor(config: ConfigPreprocessor) -> Preprocessor:
    return _run_preprocessor_cached(*_preprocessor_cache_key(config))


def _identity_cache(func):
    return func


@_identity_cache
def calculate_ftor(_preprocessor: Preprocessor,
                   well_names: List[int],
                   constraints: dict) -> CalculatorFtor:
    from frameworks_ftor.ftor.calculator import Calculator as CalculatorFtor
    from frameworks_ftor.ftor.config import Config as ConfigFtor

    config_ftor = ConfigFtor()
    # Если пользователь задал границы\значение параметра, которым производится адаптация на
    # последние точки, то эти значения применяются и для самой адаптации на последние точки
    param_last_point = config_ftor.param_name_last_points_adaptation
    if param_last_point in constraints.keys():
        if type(constraints[param_last_point]) == dict:
            config_ftor.param_bounds_last_points_adaptation = constraints[param_last_point]['bounds']
        else:
            config_ftor.apply_last_points_adaptation = False

    ftor = CalculatorFtor(
        config_ftor,
        _preprocessor.create_wells_ftor(
            well_names,
            user_constraints_for_adap_period=constraints,
        ),
        logging=False
    )
    return ftor


@_identity_cache
def calculate_wolfram(_preprocessor: Preprocessor,
                      well_names: List[int],
                      forecast_days_number: int,
                      estimator_name_group: str,
                      estimator_name_well: str,
                      is_deep_grid_search: bool,
                      window_sizes: List[int],
                      quantiles: List[float]) -> CalculatorWolfram:
    from frameworks_wolfram.wolfram.calculator import Calculator as CalculatorWolfram
    from frameworks_wolfram.wolfram.config import Config as ConfigWolfram

    wolfram = CalculatorWolfram(
        ConfigWolfram(
            forecast_days_number,
            estimator_name_group,
            estimator_name_well,
            is_deep_grid_search,
            window_sizes,
            quantiles,
        ),
        _preprocessor.create_wells_wolfram(well_names),
    )
    return wolfram


@_identity_cache
def calculate_CRM(date_start_adapt: date,
                  date_end_adapt: date,
                  date_end_forecast: date,
                  oilfield: str,
                  calc_CRM: bool = True,
                  calc_CRMIP: bool = False,
                  grad_format_data: bool = True,
                  influence_R: int = 1300,
                  maxiter: int = 100,
                  p_res: int = 220) -> CalculatorCRM or None:
    from frameworks_crm.class_CRM.calculator import Calculator as CalculatorCRM
    from frameworks_crm.class_CRM.config import ConfigCRM

    config_CRM = ConfigCRM(date_start_adapt=date_start_adapt,
                           date_end_adapt=date_end_adapt,
                           date_end_forecast=date_end_forecast,
                           calc_CRM=calc_CRM,
                           calc_CRMIP=calc_CRMIP,
                           grad_format_data=grad_format_data,
                           oilfield=oilfield)
    config_CRM.INFLUENCE_R = influence_R
    config_CRM.options_SLSQP_CRM['maxiter'] = maxiter
    config_CRM.p_res = p_res
    try:
        logger.info(f'CRM: start calculations')
        calculator_CRM = CalculatorCRM(config_CRM)
        logger.success(f'CRM: success')
        return calculator_CRM
    except Exception as exc:
        logger.exception('CRM: FAIL', exc)
        return None


@_identity_cache
def calculate_fedot(oilfield: str,
                    train_start: date,
                    train_end: date,
                    predict_start: date,
                    predict_end: date,
                    wells_norm: List,
                    coeff: pd.DataFrame,
                    lags: pd.DataFrame = None) -> CalculatorFedot:
    from frameworks_hybrid_crm_ml.class_Fedot.calculator import CalculatorFedot
    from frameworks_hybrid_crm_ml.class_Fedot.config import ConfigFedot

    config_Fedot = ConfigFedot(oilfield=oilfield,
                               train_start=train_start,
                               train_end=train_end,
                               predict_start=predict_start,
                               predict_end=predict_end,
                               wells_norm=wells_norm,
                               coeff_f=coeff,
                               lags=lags)
    calculator_fedot = CalculatorFedot(config_Fedot)
    return calculator_fedot

@_identity_cache
def calculate_shelf(oilfield: str,
                    shops: List[str],
                    wells_ois: List[int],
                    train_start: date,
                    train_end: date,
                    predict_start: date,
                    predict_end: date,
                    n_days_past: int,
                    n_days_calc_avg: int,
                    change_gtm_info: int) -> DataPostProcessorShelf:
    from frameworks_shelf_algo.class_Shelf.calculator import CalculatorShelf
    from frameworks_shelf_algo.class_Shelf.config import ConfigShelf

    config_shelf = ConfigShelf(oilfield=oilfield,
                               shops=shops,
                               wells_ois=wells_ois,
                               train_start=train_start,
                               train_end=train_end,
                               predict_start=predict_start,
                               predict_end=predict_end,
                               n_days_past=n_days_past,
                               n_days_calc_avg=n_days_calc_avg)
    results_shelf = CalculatorShelf(config_shelf)
    # results_shelf = DataPostProcessorShelf(config_shelf)
    return results_shelf


@_identity_cache
def calculate_ensemble(input_data: list[dict],
                       adaptation_days_number: int,
                       interval_probability: float,
                       draws: int,
                       tune: int,
                       chains: int,
                       target_accept: float,
                       name_of_y_true: str) -> dict[str, pd.DataFrame]:
    from models_ensemble.calculator import Calculator as CalculatorEnsemble
    from models_ensemble.config import Config as ConfigEnsemble

    calculator_ensemble = CalculatorEnsemble(
        ConfigEnsemble(adaptation_days_number=adaptation_days_number,
                       interval_probability=interval_probability,
                       draws=draws,
                       tune=tune,
                       chains=chains,
                       target_accept=target_accept,
                       name_of_y_true=name_of_y_true),
        input_data,
        logging=False
    )
    return calculator_ensemble.result_test, calculator_ensemble.weights


@cache_memo
def calculate_statistics_plots(
        statistics: dict,
        field_name: str,
        date_start: date,
        date_end: date,
        well_names: tuple,
        use_abs: bool,
        exclude_wells: list,
        bin_size: int,
        add_models: str = None,
) -> Tuple[Dict[str, go.Figure], ConfigStatistics]:
    from statistics_explorer.config import ConfigStatistics
    from statistics_explorer.main import calculate_statistics

    config_stat = ConfigStatistics(
        oilfield=field_name,
        dates=pd.date_range(date_start, date_end, freq='D').date,
        well_names=well_names,
        use_abs=use_abs,
        bin_size=bin_size,
    )
    config_stat.MODEL_NAMES[add_models] = add_models
    config_stat.exclude_wells(exclude_wells)
    analytics_plots = calculate_statistics(statistics, config_stat)
    return analytics_plots, config_stat
