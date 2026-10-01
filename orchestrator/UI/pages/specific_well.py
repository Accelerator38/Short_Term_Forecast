import datetime
from typing import Dict, Optional, List, Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from UI.app_state import AppState
from UI.cached_funcs import run_preprocessor
from statistics_explorer.config import ConfigStatistics
from statistics_explorer.plots import calc_relative_error
# from UI.pages.resume_app import external_stats

ALL_WELLS_LABEL = "Все скважины"
DF_CHESS_LIQ_COL = "Дебит жидкости"
DF_CHESS_OIL_COL = "Дебит нефти"
DF_CHESS_PRESSURE_COL = "Давление забойное"
DF_CHESS_EVENT_COL = "Мероприятие"


def show(session: st.session_state) -> None:
    state = session.state
    if not state.statistics:
        st.info('Здесь будет отображаться прогноз добычи по выбранной скважине.\n'
                'На данный момент ни одна скважина не рассчитана.\n'
                'Выберите настройки и нажмите кнопку **Запустить расчеты**.')
        return
    well_to_draw = draw_well_plot(state)
    if well_to_draw is None:
        return
    # Вывод параметров адаптации модели пьезопроводности
    if well_to_draw in state.adapt_params:
        st.write('Результаты адаптации модели пьезопроводности:', state.adapt_params[well_to_draw])
    st.info('Справка по графику:  \n'
            '**Левая часть графиков** отображает период адаптации моделей.  \n'
            '**Правая часть графиков** отображает период прогноза моделей.  \n'
            'Период адаптации отделяется от периода прогноза вертикальной пунктирной линией. '
            'На правой части графика отображается вторая вертикальная пунктирная линия, '
            'которая показывает дату начала прогноза ансамбля моделей.')


def has_preprocessor_data(config) -> bool:
    if config is None or not getattr(config, "field_name", None):
        return False
    from tools_preprocessor.preprocessor import Preprocessor

    return (Preprocessor._path_general / config.field_name).exists()


def wells_from_statistics(statistics: Dict[str, pd.DataFrame]) -> List[str]:
    wells = set()
    for df in (statistics or {}).values():
        for column in df.columns:
            parts = str(column).rsplit("_", 2)
            if len(parts) == 3 and parts[1] in {"liq", "oil", "gas"} and parts[2] in {"true", "pred"}:
                wells.add(parts[0])
    return sorted(wells)


def available_specific_wells(state: AppState, available_ois_wells: Optional[Iterable[int]] = None) -> List[str]:
    selected_wells = list(state.selected_wells_norm or [])
    well_mapping = state.wellnames_key_normal or {}
    statistics_wells = wells_from_statistics(state.statistics or {})

    if statistics_wells:
        if ALL_WELLS_LABEL in selected_wells or not selected_wells:
            return statistics_wells
        selected_statistics_wells = [well for well in selected_wells if well in statistics_wells]
        return sorted(selected_statistics_wells or statistics_wells)

    if ALL_WELLS_LABEL in selected_wells:
        wells = sorted(well_mapping)
    else:
        wells = [well for well in selected_wells if well in well_mapping]
        if not wells:
            wells = sorted(well_mapping)

    if available_ois_wells is None:
        return sorted(wells)

    available_ois_wells = set(available_ois_wells)
    return sorted(well for well in wells if well_mapping.get(well) in available_ois_wells)


def build_df_chess_from_statistics(statistics: Dict[str, pd.DataFrame], wellname: str) -> pd.DataFrame:
    for df in (statistics or {}).values():
        liq_col = f"{wellname}_liq_true"
        oil_col = f"{wellname}_oil_true"
        if liq_col not in df.columns or oil_col not in df.columns:
            continue
        return pd.DataFrame(
            {
                DF_CHESS_LIQ_COL: df[liq_col],
                DF_CHESS_OIL_COL: df[oil_col],
                DF_CHESS_PRESSURE_COL: pd.Series(index=df.index, dtype=float),
                DF_CHESS_EVENT_COL: pd.Series(index=df.index, dtype=object),
            },
            index=df.index,
        )

    return pd.DataFrame(
        columns=[DF_CHESS_LIQ_COL, DF_CHESS_OIL_COL, DF_CHESS_PRESSURE_COL, DF_CHESS_EVENT_COL]
    )


def get_df_chess_for_well(state: AppState, preprocessor, well_to_draw: str, well_name_ois) -> pd.DataFrame:
    if preprocessor is not None:
        try:
            wells_ftor = preprocessor.create_wells_ftor([well_name_ois])
            if wells_ftor:
                return wells_ftor[0].df_chess
        except AssertionError:
            pass

    return build_df_chess_from_statistics(state.statistics, well_to_draw)


def draw_well_plot(state: AppState) -> Optional[str]:
    preprocessor = None
    if not has_preprocessor_data(state.was_config):
        field_name = getattr(state.was_config, "field_name", "")
        if not state.statistics:
            st.error(f'Не найдены исходные данные для месторождения "{field_name}".')
            st.info('Выберите доступное месторождение в боковой панели и запустите расчеты заново.')
            return None
    else:
        try:
            preprocessor = run_preprocessor(state.was_config)
        except FileNotFoundError:
            preprocessor = None

    well_options = available_specific_wells(state, getattr(preprocessor, "_well_names", None))
    if not well_options:
        st.info('Нет доступных скважин для отображения. Импортированный файл не содержит рядов скважин или выбранные скважины отсутствуют в результатах.')
        return None
    if st.session_state.get('well_to_calc') not in well_options:
        st.session_state.pop('well_to_calc', None)

    well_to_draw = st.selectbox(label='Скважина',
                                options=well_options,
                                key='well_to_calc')
    well_name_ois = (state.wellnames_key_normal or {}).get(well_to_draw, well_to_draw)
    df_chess = get_df_chess_for_well(state, preprocessor, well_to_draw, well_name_ois)
    if df_chess.empty:
        st.info('Не удалось найти фактические ряды выбранной скважины в импортированных результатах.')
        return None
    fig = create_well_plot_UI(statistics=state.statistics,
                              date_test=state.was_date_test,
                              date_test_if_ensemble=state.was_date_test_if_ensemble,
                              df_chess=df_chess,
                              wellname=well_to_draw,
                              MODEL_NAMES=ConfigStatistics.MODEL_NAMES,
                              ensemble_interval=state.ensemble_interval)
    # Построение графика
    st.plotly_chart(fig, use_container_width=True)
    return well_to_draw


def create_well_plot_UI(statistics: Dict[str, pd.DataFrame],
                        date_test: datetime.date,
                        date_test_if_ensemble: datetime.date,
                        df_chess: pd.DataFrame,
                        wellname: str,
                        MODEL_NAMES: Dict[str, str],
                        ensemble_interval: pd.DataFrame = pd.DataFrame()) -> go.Figure:
    fig = make_subplots(rows=4, cols=2, shared_xaxes=True, x_title='Дата',
                        vertical_spacing=0.07,
                        horizontal_spacing=0.06,
                        column_widths=[0.65, 0.35],
                        column_titles=['', 'Прогноз'],
                        subplot_titles=['Дебит жидкости, м3/сут', '',
                                        'Дебит нефти, м3/сут', '',
                                        'Обводнённость, %', 'Относительная ошибка по нефти, %',
                                        'Забойное давление, атм', ''])
    fig.update_layout(font=dict(size=15), template='seaborn',
                      title_text=f'Скважина {wellname}',
                      legend=dict(orientation="v",
                                  font=dict(size=10),
                                  traceorder='normal'),
                      margin=dict(l=50, r=50, t=100, b=50),
                      height=630, width=1300)
    date_end_adapt = date_test - datetime.timedelta(days=1)
    df_chess = df_chess.copy().dropna(subset=[DF_CHESS_LIQ_COL, DF_CHESS_OIL_COL], how='any')
    df_chess_train = df_chess[:date_end_adapt]
    df_chess_test = df_chess[date_test:]
    statistics_train = {key: df[:date_end_adapt] for key, df in statistics.items()}
    statistics_test = {key: df[date_test:] for key, df in statistics.items()}
    ensemble_interval_train = ensemble_interval[:date_end_adapt]
    ensemble_interval_test = ensemble_interval[date_test:]
    # Адаптация
    fig = add_traces_to_specific_column(fig, statistics_train, df_chess_train,
                                        wellname, MODEL_NAMES, ensemble_interval_train,
                                        column=1, showlegend=False, marker_size=3)
    # Прогноз
    fig = add_traces_to_specific_column(fig, statistics_test, df_chess_test,
                                        wellname, MODEL_NAMES, ensemble_interval_test,
                                        column=2, showlegend=True, marker_size=3)
    calced_liq = f'{wellname}_liq_lower' in ensemble_interval.columns
    calced_oil = f'{wellname}_oil_lower' in ensemble_interval.columns
    if not ensemble_interval.empty and (calced_liq or calced_oil):
        fig.add_vline(x=date_test_if_ensemble, line_width=1, line_dash='dash', row='all', col=2)
    fig.add_vline(x=date_test, line_width=1, line_dash='dash')
    return fig


def add_traces_to_specific_column(
        fig: go.Figure,
        statistics: Dict[str, pd.DataFrame],
        df_chess: pd.DataFrame,
        wellname: str,
        MODEL_NAMES: Dict[str, str],
        ensemble_interval: pd.DataFrame,
        column: int,
        showlegend: bool,
        marker_size: int,
) -> go.Figure:
    mark, m = dict(size=marker_size), 'markers'
    colors = {'ftor': px.colors.qualitative.Pastel[1],
              'fedot': px.colors.qualitative.Pastel[2],
              'wolfram': 'rgba(248, 156, 116, 0.8)',
              'CRM': px.colors.qualitative.Pastel[6],
              'CRMIP': px.colors.qualitative.Vivid[4],
              'shelf': px.colors.qualitative.Pastel[7],
              'ensemble': 'rgba(115, 175, 72, 0.7)',
              'ensemble_interval': 'rgba(184, 247, 212, 0.7)',
              'true': 'rgba(99, 110, 250, 0.7)',
              'pressure': '#C075A6'}
    y_liq_true = df_chess[DF_CHESS_LIQ_COL]
    y_oil_true = df_chess[DF_CHESS_OIL_COL]
    # Доверительный интервал ансамбля
    if not ensemble_interval.empty:
        if f'{wellname}_liq_lower' in ensemble_interval.columns:
            trace = go.Scatter(name=f'Доверит. интервал',
                               x=ensemble_interval.index, y=ensemble_interval[f'{wellname}_liq_lower'],
                               mode='lines', line=dict(width=1, color=colors['ensemble_interval']),
                               showlegend=showlegend,
                               legendgroup=f'group3_{ensemble_interval.columns}')
            fig.add_trace(trace, row=1, col=column)
            trace = go.Scatter(name=f'LIQ: Доверит. интервал',
                               x=ensemble_interval.index, y=ensemble_interval[f'{wellname}_liq_upper'],
                               fill='tonexty', mode='lines', line=dict(width=1, color=colors['ensemble_interval']),
                               showlegend=False,
                               legendgroup=f'group3_{ensemble_interval.columns}')
            fig.add_trace(trace, row=1, col=column)
        if f'{wellname}_oil_lower' in ensemble_interval.columns:
            trace = go.Scatter(name=f'OIL: Доверит. интервал',
                               x=ensemble_interval.index, y=ensemble_interval[f'{wellname}_oil_lower'],
                               mode='lines', line=dict(width=1, color=colors['ensemble_interval']),
                               showlegend=False,
                               legendgroup=f'group3_{ensemble_interval.columns}')
            fig.add_trace(trace, row=2, col=column)
            trace = go.Scatter(name=f'OIL: Доверит. интервал',
                               x=ensemble_interval.index, y=ensemble_interval[f'{wellname}_oil_upper'],
                               fill='tonexty', mode='lines', line=dict(width=1, color=colors['ensemble_interval']),
                               showlegend=False,
                               legendgroup=f'group3_{ensemble_interval.columns}')
            fig.add_trace(trace, row=2, col=column)
    # Факт
    trace = go.Scatter(name=f'{MODEL_NAMES["true"]}', x=y_liq_true.index, y=y_liq_true,
                       mode=m, marker=dict(size=5, color=colors['true']), showlegend=showlegend)
    fig.add_trace(trace, row=1, col=column)
    trace = go.Scatter(name=f'OIL: {MODEL_NAMES["true"]}', x=y_oil_true.index, y=y_oil_true,
                       mode=m, marker=dict(size=5, color=colors['true']), showlegend=False)
    fig.add_trace(trace, row=2, col=column)
    if column == 1:
        # Обводнённость
        trace_obv = go.Scatter(name='Обводнённость', x=y_liq_true.index, y=((1 - y_oil_true / y_liq_true) * 100),
                               mode=m, marker=dict(size=5, color='#19D3F3'), showlegend=True)
        fig.add_trace(trace_obv, row=3, col=column)
    # Прогнозы моделей
    for model in statistics:
        if f'{wellname}_oil_pred' in statistics[model]:
            clr = colors[model]
            y_liq = statistics[model][f'{wellname}_liq_pred'].dropna()
            y_oil = statistics[model][f'{wellname}_oil_pred'].dropna()
            trace_liq = go.Scatter(name=f'{MODEL_NAMES[model]}', x=y_liq.index, y=y_liq,
                                   mode=m, marker=mark, line=dict(width=1, color=clr),
                                   showlegend=showlegend,
                                   legendgroup=f'group_{model}')

            fig.add_trace(trace_liq, row=1, col=column)  # Дебит жидкости
            trace_oil = go.Scatter(name=f'OIL: {MODEL_NAMES[model]}', x=y_oil.index, y=y_oil,
                                   mode=m, marker=mark, line=dict(width=1, color=clr),
                                   showlegend=False,
                                   legendgroup=f'group_{model}')
            fig.add_trace(trace_oil, row=2, col=column)  # Дебит нефти
            deviation = calc_relative_error(y_oil_true, y_oil, use_abs=False)
            trace_err = go.Scatter(name=f'OIL ERR: {MODEL_NAMES[model]}', x=deviation.index, y=deviation,
                                   mode=m, marker=dict(size=4), line=dict(width=1, color=clr),
                                   showlegend=False,
                                   legendgroup=f'group_{model}')
            fig.add_trace(trace_err, row=3, col=column)  # Ошибка по нефти
    # Забойное давление
    pressure = df_chess[DF_CHESS_PRESSURE_COL]
    trace_pressure = go.Scatter(name=f'Заб. давление', x=pressure.index, y=pressure,
                                mode=m, marker=dict(size=4, color=colors['pressure']),
                                showlegend=showlegend,
                                legendgroup=f'group1_{model}')
    fig.add_trace(trace_pressure, row=4, col=column)
    fig.update_layout(
        legend=dict(
            # orientation="h",
            font=dict(size=10))
    )
    # Мероприятия
    events = df_chess[DF_CHESS_EVENT_COL]
    _events = events.dropna()
    trace_events = go.Scatter(name='Мероприятие', x=_events.index, y=[0.2] * len(_events),
                              mode='markers+text', marker=dict(size=5, color='#AB63FA'), text=_events.array,
                              textposition='top center', textfont=dict(size=12),
                              showlegend=True,
                              legendgroup=f'group2_{model}')
    fig.update_layout(
        legend=dict(
            # orientation="h",
            font=dict(size=10))
    )
    fig.add_trace(trace_events, row=4, col=column)
    return fig
