import datetime as dt
from pathlib import Path
import pandas as pd
from frameworks_shelf_algo.class_Shelf.constants import DATA, DATE_FORMAT, DEC_RATES, DEC_RATES_LIQ, DEC_RATES_GAS


def get_date_range(_date_start: dt.date, _date_end: dt.date) -> list:
    return [*map(lambda time_stamp: time_stamp.date(), pd.date_range(_date_start, _date_end))]


def str_to_datetime(date_str: str) -> dt.date:
    return dt.datetime.strptime(date_str, DATE_FORMAT).date()


def datetime_to_str(date_datetime: dt.date) -> str:
    return date_datetime.strftime(DATE_FORMAT)


def _get_path(oilfield):
    project_root = Path(__file__).parent.parent.parent
    return project_root / 'tools_preprocessor' / 'data' / f'{oilfield}'


# темп падения для нефти
def get_s_decline_rates(_data: dict, _well: str, _date_start: dt.date, _date_end: dt.date) -> pd.Series:
    dates = get_date_range(_date_start, _date_end)
    s_dec_rates = pd.Series(index=dates, dtype=float)
    for date, val in sorted(_data[_well][DEC_RATES].items()):
        s_dec_rates[s_dec_rates.index >= date] = val
    return s_dec_rates


# темп падения для жидкости
def get_s_decline_rates_liq(_data: dict, _well: str, _date_start: dt.date, _date_end: dt.date) -> pd.Series:
    dates = get_date_range(_date_start, _date_end)
    s_dec_rates_liq = pd.Series(index=dates, dtype=float)
    for date, val in sorted(_data[_well][DEC_RATES_LIQ].items()):
        s_dec_rates_liq[s_dec_rates_liq.index >= date] = val
    return s_dec_rates_liq


def get_s_decline_rates_gas(_data: dict, _well: str, _date_start: dt.date, _date_end: dt.date) -> pd.Series:
    dates = get_date_range(_date_start, _date_end)
    s_dec_rates_gas = pd.Series(index=dates, dtype=float)
    decline_rates_gas = _data[_well].get(DEC_RATES_GAS, {})
    if not decline_rates_gas:
        s_dec_rates_gas[:] = 0.0
        return s_dec_rates_gas
    for date, val in sorted(decline_rates_gas.items()):
        s_dec_rates_gas[s_dec_rates_gas.index >= date] = val
    return s_dec_rates_gas


def transform_str_dates_to_datetime_or_vice_versa(dict_: dict, dates_to_datetime: bool) -> None:
    keys = [*dict_.keys()]
    for key in keys:
        val = dict_[key]
        was_transformed, key_transformed = transform_if_possible(key, dates_to_datetime)
        if was_transformed:
            dict_[key_transformed] = val
            del dict_[key]
        was_transformed, val_transformed = transform_if_possible(val, dates_to_datetime)
        if isinstance(val_transformed, dict):
            transform_str_dates_to_datetime_or_vice_versa(val_transformed, dates_to_datetime)
        elif was_transformed:
            dict_[key_transformed] = val_transformed


def transform_if_possible(val, dates_to_datetime):
    was_transformed = False
    if dates_to_datetime:
        if isinstance(val, str):
            try:
                val = str_to_datetime(val)
                was_transformed = True
            except ValueError:
                pass
    else:
        if isinstance(val, dt.date):
            val = datetime_to_str(val)
            was_transformed = True
    return was_transformed, val
