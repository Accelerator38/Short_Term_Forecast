import datetime
from pathlib import Path
import json
import pandas as pd
import numpy as np
from tools_preprocessor.config import Config as ConfigPreprocessor
from tools_preprocessor.preprocessor import Preprocessor

DEBIT = 'Дебит нефти расчетный'
STATUS = 'sost'


def datetime_to_str(date_datetime: datetime.date) -> str:
    return date_datetime.strftime('%Y-%m-%d')


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


if __name__ == '__main__':
    field_name = 'Отдельное'
    shops = ['ЦДHГ-1']
    n_days_past = 30
    n_days_calc_avg = 5
    date_test = datetime.date(2022, 4, 1)
    date_end = datetime.date(2022, 4, 30)

    path = Path.cwd() / 'tools_preprocessor' / 'data' / field_name / 'sh_sost_fond.feather'
    df_sh_sost_fond = pd.read_feather(path)
    df_sh_sost_fond.set_index('dt', inplace=True)
    date_start = df_sh_sost_fond.index[0]
    preprocessor = Preprocessor(
        ConfigPreprocessor(
            field_name,
            shops,
            date_start,
            date_test,
            date_end,
        )
    )
    df_fact_test_prd = pd.DataFrame()
    data_shelf = dict()
    data_shelf['Плановые остановы МЛСП'] = dict()
    for well in preprocessor.well_names:
        df_well = df_sh_sost_fond[df_sh_sost_fond['well.ois'] == well]
        df_fact_test_prd[well] = df_well[DEBIT][(date_test <= df_well.index) & (df_well.index <= date_end)]
        df_well_work_before_test = df_well[(df_well[STATUS] == 'В работе') & (df_well.index < date_test)]
        avg_debit_in_past = np.mean(df_well_work_before_test[DEBIT][-n_days_past:-n_days_past + n_days_calc_avg])
        avg_debit_before_test = np.mean(df_well_work_before_test[DEBIT][-n_days_calc_avg:])
        dec_rate = (avg_debit_in_past - avg_debit_before_test) / (n_days_past - n_days_calc_avg)
        data_shelf[well] = {
            'последний замер': {
                'дата': datetime_to_str(df_well_work_before_test.index[-1]),
                'значение': df_well_work_before_test[DEBIT].iloc[-1]},
            'темпы падения': {
                datetime_to_str(date_test): dec_rate},
            'гтмы': dict(),
        }
        for stop_prd_start_date, stop_prd_end_date in get_stop_prd_dates(df_well):
            data_shelf[well]['гтмы'][datetime_to_str(stop_prd_start_date)] = {
                'название': 'Текущий ремонт скважин',
                'дебит в период ТРС': 0,
                'длительность ТРС': (stop_prd_end_date - stop_prd_start_date).days + 1,
                'длительность выхода на режим': 1
            }
    with open(f"data_{field_name}.json", "w", encoding='UTF-8') as outfile:
        json.dump(data_shelf, outfile, ensure_ascii=False)
    df_fact_test_prd.to_excel('fact_test_period.xlsx')

