import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import n_colors
import streamlit as st
from typing import Tuple, List, Optional

from UI.pages.analytics import select_wells_set, draw_form_exclude_wells
from statistics_explorer.plots import calc_relative_error
from statistics_explorer.config import ConfigStatistics
from UI.app_state import AppState

MODE_LABELS = {'Нефть': 'oil', 'Жидкость': 'liq', 'Газ': 'gas'}


def show(session: st.session_state) -> None:
    state = session.state
    if not state.statistics:
        st.info('Здесь будет отображаться карта скважин, выбранных для расчета.\n'
                'На данный момент ни одна скважина не рассчитана.\n'
                'Выберите настройки и нажмите кнопку **Запустить расчеты**.')
        return
    selected_wells_set = select_wells_set(state)
    fig, selected_plot, fig_table = select_plot(state, selected_wells_set)
    st.plotly_chart(fig, use_container_width=True)
    if selected_plot == 'Карта скважин':
        st.plotly_chart(fig_table, use_container_width=True)
    if selected_plot == 'TreeMap':
        draw_form_exclude_wells(state, selected_wells_set)
        st.info('Справка по TreeMap:  \n'
                '**Цвет сектора** зависит от средней посуточной ошибки на периоде прогноза (модуль отклонения).  \n'
                '**Размер сектора** зависит от накопленной добычи на периоде прогноза, [м3].  \n'
                'Надписи внутри каждого сектора идут в следующем порядке:'
                '  \n- имя скважины,  \n- накопленная добыча,  \n- посуточная ошибка на прогнозе.  \n')


def has_crm_map_data(state: AppState) -> bool:
    coeff_f = state.coeff_f
    coords = state.wells_coords_CRM
    return (
        isinstance(coeff_f, pd.DataFrame)
        and isinstance(coords, pd.DataFrame)
        and not coeff_f.empty
        and not coords.empty
    )


def available_plot_options(state: AppState) -> List[str]:
    options = ['TreeMap']
    if has_crm_map_data(state):
        options.insert(0, 'Карта скважин')
    return options


def select_plot(state: AppState, selected_wells_set: Tuple[str, ...]) -> [go.Figure, str]:
    selected_plot = st.selectbox(label='', options=available_plot_options(state))
    if selected_plot != 'TreeMap':
        mode_well = select_well(state.coeff_f.columns)
        state.coeff_f = state.coeff_f / state.coeff_f.sum()
        coords_df, f_dict = crm_map(state.coeff_f, state.wells_coords_CRM)
        df_f_coeff = plot_influence_table(state.coeff_f, mode_well)
        return crm_plot(coords_df, f_dict, mode_well, state.CRM_influence_R), selected_plot, df_f_coeff

    mode_options = available_treemap_mode_labels(state, selected_wells_set)
    if not mode_options:
        st.info('Нет данных для построения TreeMap. В результатах нет пар колонок *_true / *_pred для выбранных скважин.')
        return create_empty_tree_plot('Нет данных для TreeMap'), selected_plot, None

    mode = st.selectbox(label='Жидкость/нефть/газ', options=mode_options)
    mode_code = MODE_LABELS[mode]
    model_for_error = select_model(state, mode_code, selected_wells_set)
    if model_for_error is None:
        st.info('Нет данных для построения TreeMap по выбранному режиму. Выберите другой режим или модель с рассчитанными колонками.')
        return create_empty_tree_plot('Нет данных для TreeMap'), selected_plot, None
    df = prepare_data_for_treemap(state, model_for_error, selected_wells_set, mode=mode_code)
    if df.empty:
        st.info('Для выбранной модели нет скважин с полным набором колонок для этого режима.')
        return create_empty_tree_plot('Нет скважин с данными'), selected_plot, None
    return create_tree_plot(df, mode=mode), selected_plot, None


def has_treemap_mode_data(state: AppState, model: str, mode: str, selected_wells_set: Tuple[str, ...]) -> bool:
    model_statistics = state.statistics.get(model, pd.DataFrame())
    if model_statistics.empty:
        return False

    well_names = [
        elem for elem in selected_wells_set if elem not in state.exclude_wells
    ]
    for well in well_names:
        true_col = f'{well}_{mode}_true'
        pred_col = f'{well}_{mode}_pred'
        if true_col not in model_statistics.columns or pred_col not in model_statistics.columns:
            continue
        df_test_period = test_period_statistics(model_statistics, state.was_date_test)
        if not df_test_period.empty and not df_test_period[[true_col, pred_col]].dropna(how='all').empty:
            return True
    return False


def available_treemap_mode_labels(state: AppState, selected_wells_set: Tuple[str, ...]) -> List[str]:
    return [
        label
        for label in sorted(MODE_LABELS)
        if available_treemap_models(state, MODE_LABELS[label], selected_wells_set)
    ]


def available_treemap_models(state: AppState, mode: str, selected_wells_set: Tuple[str, ...]) -> List[str]:
    excluded_models = {'true'}
    if mode in {'oil', 'gas'}:
        excluded_models.add('CRM')
    return [
        model
        for model in state.statistics
        if model not in excluded_models and has_treemap_mode_data(state, model, mode, selected_wells_set)
    ]


def select_model(state: AppState, mode: str, selected_wells_set: Tuple[str, ...]) -> Optional[str]:
    MODEL_NAMES = ConfigStatistics.MODEL_NAMES
    MODEL_NAMES_REVERSED = {v: k for k, v in MODEL_NAMES.items()}
    model_keys = available_treemap_models(state, mode, selected_wells_set)
    if not model_keys:
        return None
    models_without_ensemble = [MODEL_NAMES.get(model, model) for model in model_keys]
    model = st.selectbox(label="Модель для расчета ошибки:",
                         options=models_without_ensemble)
    return MODEL_NAMES_REVERSED.get(model, model)


def prepare_data_for_treemap(state: AppState, model: str, selected_wells_set: Tuple[str, ...], mode: str = 'liq') -> pd.DataFrame:
    columns = ['wellname', 'cum_q_liq', 'cum_q_oil', 'cum_q_gas', 'err_liq', 'err_oil', 'err_gas']
    df = pd.DataFrame(columns=columns)
    model_statistics = state.statistics.get(model, pd.DataFrame())
    well_names = [
        elem for elem in selected_wells_set if elem not in state.exclude_wells
    ]
    for well in well_names:
        true_col = f'{well}_{mode}_true'
        pred_col = f'{well}_{mode}_pred'
        if true_col not in model_statistics.columns or pred_col not in model_statistics.columns:
            continue

        df_test_period = test_period_statistics(model_statistics, state.was_date_test)
        if df_test_period.empty:
            continue

        row = {
            'wellname': well,
            'cum_q_liq': None,
            'cum_q_oil': None,
            'cum_q_gas': None,
            'err_liq': None,
            'err_oil': None,
            'err_gas': None,
        }
        row[f'cum_q_{mode}'] = round(df_test_period[true_col].sum(), 1)
        err = calc_relative_error(df_test_period[true_col],
                                  df_test_period[pred_col],
                                  use_abs=True)
        row[f'err_{mode}'] = round(err.mean(), 1)
        df.loc[len(df)] = row
    return df


def test_period_statistics(df: pd.DataFrame, date_test) -> pd.DataFrame:
    if df.empty or date_test is None:
        return df
    try:
        sliced = df[date_test:]
        if not sliced.empty:
            return sliced
    except Exception:
        pass

    converted = df.copy()
    try:
        converted.index = pd.to_datetime(converted.index)
        sliced = converted[pd.Timestamp(date_test):]
        if not sliced.empty:
            return sliced
    except Exception:
        pass

    return df


def create_tree_plot(df: pd.DataFrame, mode: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(font=dict(size=15),
                      title_text=f'Treemap',
                      height=630,
                      width=1300,
                      separators='. ')
    mode_dict = {'Нефть': 'oil', 'Жидкость': 'liq', 'Газ': 'gas'}
    mode = mode_dict[mode]
    df['text_error'] = 'Ошибка: ' + df[f'err_{mode}'].apply(str) + '%'
    fig.add_trace(go.Treemap(labels=df['wellname'],
                             parents=["Все скважины" for _ in df['wellname']],
                             values=df[f'cum_q_{mode}'],
                             textinfo="text+label+value",
                             text=df['text_error'],
                             **{'marker_cmin': 0,
                                'marker_cmax': 100,
                                'marker_colors': df[f'err_{mode}'],
                                'marker_colorscale': 'oranges'},
                             ))
    return fig


def create_empty_tree_plot(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title_text='Treemap',
                      height=630,
                      width=1300)
    fig.add_annotation(text=message,
                       x=0.5,
                       y=0.5,
                       xref='paper',
                       yref='paper',
                       showarrow=False,
                       font=dict(size=18))
    return fig

def select_well(produced_wells: list) -> str:
    well_chosen = st.selectbox(label="Скважина:",
                               options=list(sorted(produced_wells)), #['', 'Все скважины'] +
                               key='selected_wells_for_influence')
    return well_chosen

def crm_map(f_values: pd.DataFrame, coords_df: pd.DataFrame, border: float = 0.) -> [pd.DataFrame, dict]:
    '''
    f_values - матрица взаимовлияния
       |
       |
       v
    Dataframe n x m, n - строки, названия нагнетательных скважин:str, m - столбцы, названия добывающих скважин:str

    coords_df - координаты
       |
       |
       v
    Dataframe с 3 колонками: 'Скважина':str, 'Координата X':float, 'Координата Y':float

    prod_df - добывающие скважины
       |
       |
       v
    Dataframe с 3 колонками: 'Скважина':str, 'Дата':pd.Timestamp, 'Дебит нефти':float

    inj_df - нагнетательные скважины
       |
       |
       v
    Dataframe с 3 колонками: 'Скважина':str, 'Дата':pd.Timestamp, 'Приемистость':float

    mult - регулирует толщину линий на карте
    '''

    def wut_trans(x):
        l = [str(i) for i in x.values]
        l.sort()
        return tuple(l)

    coords_df = coords_df[['Координата X', 'Координата Y']].reset_index()
    f_values = f_values[list(set(f_values.columns) - set(f_values.index))]
    inj_wells_list = f_values.index.tolist()
    prod_wells_list = f_values.columns.tolist()
    # !!!!!!
    # f_values = f_values[f_values > 0]

    f_values = f_values[f_values > border]

    f_values = f_values + 1.5
    f_values = f_values.stack().reset_index()
    f_values['wut'] = f_values.apply(lambda x: wut_trans(x), axis=1)
    f_values = f_values.drop_duplicates(subset='wut', keep='first').drop('wut', axis=1)
    f_dict = f_values.set_index(['level_0', 'Well']).to_dict()[0]


    coords_df['Тип'] = coords_df['Скважина'].apply(lambda x: 'Добывающая' if x in prod_wells_list else \
        ('Нагнетательная' if x in inj_wells_list else np.NaN))
    coords_df['value'] = 0.

    return coords_df, f_dict


def crm_plot(coords_df: pd.DataFrame, f_dict: dict, mode:str, influence_R:int) -> go.Figure:
    _df = pd.DataFrame(columns=['x', 'y', 'value'])
    count = 0
    fig = go.Figure()

    ht_inj, m_inj, ht_prod, m_prod = None, None, None, None

    fig.add_trace(go.Scatter(
        x=coords_df[coords_df['Тип'] == 'Нагнетательная']['Координата X'].tolist(),
        y=coords_df[coords_df['Тип'] == 'Нагнетательная']['Координата Y'].tolist(),
        mode='markers',
        name='Нагнетательная скважина',
        text=coords_df[coords_df['Тип'] == 'Нагнетательная']['Скважина'].tolist(),
        textposition='top center',
        hovertext=ht_inj,
        hoverinfo='text',
        marker=m_inj,
        showlegend=True, ))

    fig.add_trace(go.Scatter(
        x=coords_df[coords_df['Тип'] == 'Добывающая']['Координата X'].tolist(),
        y=coords_df[coords_df['Тип'] == 'Добывающая']['Координата Y'].tolist(),
        mode='markers',
        name='Добывающая скважина',
        text=coords_df[coords_df['Тип'] == 'Добывающая']['Скважина'].tolist(),
        textposition='top center',
        hovertext=ht_prod,
        hoverinfo='text',
        marker=m_prod,
        showlegend=True))
    influence_count = 0
    for key, value in f_dict.items():
        if key[1] == mode:
            influence_count +=1
    influence_count += 1 if influence_count == 1 else 0
    colorscale = n_colors('rgb(0, 0, 0)', 'rgb(0, 255, 0)', influence_count, colortype='rgb')
    for key, value in f_dict.items():
        if key[1] == mode:
            _x = [coords_df[coords_df['Скважина'] == key[0]]['Координата X'].values[0],
                  coords_df[coords_df['Скважина'] == key[1]]['Координата X'].values[0]]
            _y = [coords_df[coords_df['Скважина'] == key[0]]['Координата Y'].values[0],
                  coords_df[coords_df['Скважина'] == key[1]]['Координата Y'].values[0]]

            fig.add_trace(go.Scatter(x=_x, y=_y, mode='lines',
                                     line=go.scatter.Line(color=colorscale[count], width=value),
                                     showlegend=False))
            _df.loc[count] = [np.mean(_x), np.mean(_y), value - 1.5]
            count += 1

    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  fillcolor="rgb(255,0,0)",
                  x0=coords_df[coords_df['Скважина'] == mode]['Координата X'].values[0] - influence_R,
                  y0=coords_df[coords_df['Скважина'] == mode]['Координата Y'].values[0] - influence_R,
                  x1=coords_df[coords_df['Скважина'] == mode]['Координата X'].values[0] + influence_R,
                  y1=coords_df[coords_df['Скважина'] == mode]['Координата Y'].values[0] + influence_R,
                  opacity=0.1,
                  line_color="rgb(255,0,0)",
                  )

    fig.add_trace(go.Scatter(
        x=_df['x'].tolist(),
        y=_df['y'].tolist(),
        mode='markers',
        opacity=0.2,
        text=[round(i, 3) for i in _df['value'].tolist()],
        hovertext=_df['value'].tolist(),
        hoverinfo='text',
        textposition='top center',
        marker=dict(color='black', size=5, opacity=.6),
        showlegend=False))

    fig.update_layout(
        # title='Взаимовлияние скважин',
        xaxis_title='X координата',
        yaxis_title='Y координата',
        height=630,
        width=1300,
        legend = dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="right",
        x=1,
        bgcolor='rgba(0,0,0,0)'))


    return fig

def plot_influence_table(df_table: pd.DataFrame, mode: str) -> go.Figure:
    # TODO: выводить таблицу целиком
    # if mode == "":
    #     return None
    # elif mode == "Все скважины":
    #     mode = df.columns.values()
    df_table = df_table[df_table>0].round(3)
    df_table.index = df_table.index.astype(str)
    inj_wells = []
    for w in df_table[mode].dropna().index:
        inj_wells.append(f'<b>{w}<b>')
    fig = go.Figure(data=[go.Table(header=dict(values=['', f'<b>{mode}<b>']),
                                   cells=dict(values=[inj_wells, df_table[mode].dropna()],
                                              align='center', font=dict(color='black')),
                                   )
                          ])
    return fig
