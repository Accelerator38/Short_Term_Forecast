import datetime
import os
import warnings
from pathlib import Path
from timeit import default_timer

import pandas as pd
import plotly
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

from frameworks_ftor.ftor.calculator import Calculator as CalculatorFtor
from frameworks_ftor.ftor.config import Config as ConfigFtor
from frameworks_ftor.ftor.well import Well as WellFtor
from tools_preprocessor.config import Config as ConfigPreprocessor
from tools_preprocessor.preprocessor import Preprocessor

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None
logger.remove()


def _create_trans_plot(well_name, df_chess, rates, date_test, adap_and_fixed_params, path, is_liq):
    name = 'liq' if is_liq else 'oil'
    figure = go.Figure(layout=go.Layout(
        font=dict(size=10),
        hovermode='x',
        template='seaborn',
        title=dict(text=f'{well_name} {name}', x=0.05, xanchor='left'),
    ))
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.05, 0.6, 0.35], figure=figure
    )
    fig.update_xaxes(dtick='M1', tickformat="%b\n%Y")
    fig.update_yaxes(row=1, col=1, range=[0, 1], showticklabels=False)

    m = 'markers'
    ml = 'markers+lines'
    mark = dict(size=3)
    line = dict(width=1)

    s = df_chess[WellFtor.NAME_EVENT].dropna()
    trace_a = go.Scatter(
        name=WellFtor.NAME_EVENT,
        x=s.index,
        y=[0.2] * len(s),
        mode='markers+text',
        marker=dict(size=8),
        text=s.array,
        textposition='top center',
        textfont=dict(size=8),
    )
    fig.add_trace(trace_a, row=1, col=1)

    x = df_chess.index.to_list()
    if is_liq:
        dir_path = path / 'liq'
        x_model = x
        name_fact = WellFtor.NAME_RATE_LIQ
        name_model = 'Дебит жидкости модельный'
        name_graph = 'liq'
    else:
        dir_path = path / 'oil'
        x_model = rates.index.to_list()
        name_fact = WellFtor.NAME_RATE_OIL
        name_model = 'Дебит нефти модельный'
        name_graph = 'oil'

    fig.add_trace(go.Scatter(name=name_fact, x=x, y=df_chess[name_fact], mode=m, marker=mark), row=2, col=1)
    fig.add_trace(go.Scatter(name=name_model, x=x_model, y=rates, mode=ml, marker=mark, line=line), row=2, col=1)
    fig.add_trace(go.Scatter(name='Pз', x=x, y=df_chess[WellFtor.NAME_PRESSURE], mode=m, marker=mark), row=3, col=1)
    fig.add_vline(x=date_test, line_width=2, line_dash='dash')

    text = 'params:'
    for ad_prd_prms in adap_and_fixed_params.copy():
        text += '<br>' + str({name: round(ad_prd_prms[name], 1) for name in ad_prd_prms})

    fig.add_annotation(showarrow=False, text=text, xref='paper', yref='paper', x=0.5, y=1.175)

    if not dir_path.exists():
        os.mkdir(dir_path)
    file = f'{str(dir_path)}\\{well_name} {name_graph}.png'
    plotly.io.write_image(fig, file=file, width=1450, height=700, scale=2, engine='kaleido')


if __name__ == '__main__':
    fields = [
        'Крайнее',
        # 'Вынгаяхинское',
        # 'Оренбургское',
    ]

    shops_lst = [
        ['ЦДНГ-4'],
        # ['ЦДНГ-10'],
        # ['ЦДНГ-1'],
    ]

    dates_start = [
        datetime.date(2018, 1, 1),
        # datetime.date(2018, 1, 1),
        # datetime.date(2020, 7, 31),
    ]

    dates_test = [
        datetime.date(2018, 11, 1),
        # datetime.date(2019, 1, 1),
        # datetime.date(2021, 5, 1),
    ]

    dates_end = [
        datetime.date(2019, 1, 31),
        # datetime.date(2019, 3, 31),
        # datetime.date(2021, 7, 31),
    ]

    data = zip(fields, shops_lst, dates_start, dates_test, dates_end)

    for field_name, shops, date_start, date_test, date_end in data:
        time_calc_started = str(datetime.datetime.now()).replace(':', '-')
        start = default_timer()
        name_dir = field_name + " " + time_calc_started
        path = Path.cwd() / 'tests' / name_dir
        if not path.exists():
            os.mkdir(path)

        preprocessor = Preprocessor(
            ConfigPreprocessor(
                field_name,
                shops,
                date_start,
                date_test,
                date_end,
            )
        )

        well_names = preprocessor.well_names
        well_names = [well_name for well_name in well_names if well_name in [
            2560500100,
        ]]

        data_preprocessor_lst = preprocessor.create_wells_ftor(well_names)
        calculator_ftor = CalculatorFtor(
            ConfigFtor(),
            data_preprocessor_lst,
            logging=False,
        )
        wells_ftor = calculator_ftor.wells

        for well_ftor in wells_ftor:
            well_name = well_ftor.well_name
            data_preprocessor = None
            try:
                for data in data_preprocessor_lst:
                    if data.well_name == well_name:
                        data_preprocessor = data
                        break

                df_chess = data_preprocessor.df_chess
                res_ftor = well_ftor.results

                adap_and_fixed_params = res_ftor.adap_and_fixed_params
                rates_liq_ftor = pd.concat(objs=[res_ftor.rates_liq_train, res_ftor.rates_liq_test])
                rates_oil_ftor = res_ftor.rates_oil_test

                df = pd.DataFrame()
                df[f'{well_name}_oil_true'] = df_chess['Дебит нефти'].loc[df_chess.index >= date_test]
                df[f'{well_name}_oil_pred'] = rates_oil_ftor
                df[f'{well_name}_liq_true'] = df_chess['Дебит жидкости'].loc[df_chess.index >= date_test]
                df[f'{well_name}_liq_pred'] = rates_liq_ftor.loc[df_chess.index >= date_test]
                df.to_excel(path / f'{well_name}.xlsx')

                _create_trans_plot(well_name, df_chess, rates_liq_ftor, date_test, adap_and_fixed_params, path, is_liq=True)
                _create_trans_plot(well_name, df_chess, rates_oil_ftor, date_test, adap_and_fixed_params, path, is_liq=False)
            except Exception as exc:
                file = open(path / f'{well_name} error.txt', 'w')
                file.write(str(exc))
                file.close()

        exec_time = default_timer() - start
        file = open(path / 'test_data.txt', 'w')
        print(f'{time_calc_started = }', file=file)
        print(f'{exec_time = } с', file=file)
        print(f'{field_name = }', file=file)
        print(f'{shops = }', file=file)
        print(f'{date_start = }', file=file)
        print(f'{date_test = }', file=file)
        print(f'{date_end = }', file=file)
        file.close()

        results_table = path / 'aggregated_results.xlsx'
        well_tables = {*path.glob('*.xlsx')}
        df_results = pd.DataFrame()
        for table in well_tables:
            df = pd.read_excel(table, index_col='dt')
            for col in df:
                df_results[col] = df[col]
        df_results.to_excel(results_table)
