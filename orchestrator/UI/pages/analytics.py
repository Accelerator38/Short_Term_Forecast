from typing import Tuple
import pandas as pd
import streamlit as st

from UI.app_state import AppState
from UI.cached_funcs import calculate_statistics_plots


def show(session: st.session_state) -> None:
    state = session.state
    if not state.statistics_test_only:
        st.info('Здесь будет отображаться статистика по выбранному набору скважин.')
        return
    if state.statistics_another_models:
        selected_wells_set = select_wells_set(state)
        draw_statistics_plots(state, selected_wells_set, state.statistics_another_models)
        draw_form_exclude_wells(state, selected_wells_set)
    else:
        selected_wells_set = select_wells_set(state)
        draw_statistics_plots(state, selected_wells_set, 'str')
        draw_form_exclude_wells(state, selected_wells_set)


def select_wells_set(state: AppState) -> Tuple[str, ...]:
    wells_in_model = []
    for df in state.statistics_test_only.values():
        wells_in_model.append(set([col.split('_')[0] for col in df.columns]))
    # Можно строить статистику либо для общего набора скважин (скважина рассчитана всеми моделями),
    # либо для всех скважин (скважина рассчитана хотя бы одной моделью).
    # Выберите, что подать в конфиг ниже: well_names_common или well_names_all.
    well_names_all = tuple(set.union(*wells_in_model))
    well_names_common = tuple(set.intersection(*wells_in_model))
    well_names_for_statistics = well_names_all
    return well_names_for_statistics


def draw_statistics_plots(state: AppState, selected_wells_set: Tuple[str, ...], add_models: str) -> None:
    analytics_plots, config_stat = calculate_statistics_plots(
        statistics=state.statistics_test_only,
        field_name=state.was_config.field_name,
        date_start=state.statistics_test_index[0],
        date_end=state.statistics_test_index[-1],
        well_names=selected_wells_set,
        use_abs=True,
        exclude_wells=state.exclude_wells,
        bin_size=10,
        add_models=add_models
    )
    available_plots = [plot_name for plot_name in analytics_plots if plot_name not in config_stat.ignore_plots]
    plots_mode = select_plots_subset()
    plots_to_draw = [plot_name for plot_name in available_plots if plots_mode in plot_name]
    stat_to_draw = st.selectbox(label='Выбор графика:',
                                options=reversed(sorted(plots_to_draw)),
                                key='stat_to_draw')
    st.plotly_chart(analytics_plots[stat_to_draw], use_container_width=True)
    st.plotly_chart(analytics_plots["Статистика по моделям"], use_container_width=True)


def select_plots_subset() -> str:
    MODES = {'Жидкость': 'жидк', 'Нефть': 'нефт', 'Газ': 'газ'}
    mode = st.selectbox(label='Жидкость/нефть/газ',
                        options=MODES)
    return MODES[mode]


def draw_form_exclude_wells(state: AppState, selected_wells_set: Tuple[str, ...]) -> None:
    # Форма "Исключить скважины из статистики"
    form = st.form("form_exclude_wells")
    form.multiselect("Исключить скважины из статистики:",
                     options=sorted(selected_wells_set),
                     default=sorted(state.exclude_wells),
                     key="mselect_exclude_wells")
    form.form_submit_button("Применить", on_click=update_exclude_wells, args=(state,))


def update_exclude_wells(state: AppState) -> None:
    state.exclude_wells = st.session_state.mselect_exclude_wells
