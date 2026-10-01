import io
import pickle
import datetime as dt
from pathlib import Path
from typing import IO, Mapping, Any, Optional

import pandas as pd
import streamlit as st
from loguru import logger

from UI.app_state import AppState
from UI.config import FIELDS_SHOPS
from statistics_explorer.config import ConfigStatistics


def show(session: st.session_state) -> None:
    state = session.state
    draw_upload_state(state, session)
    draw_export_state(state)
    draw_export_excel(state)
    draw_upload_oilfield_data()
    external_stats(state)

def draw_upload_state(state: AppState, session: Optional[Any] = None) -> None:
    st.subheader("Импорт готового состояния программы")
    st.write(
        "При импорте приложение попытается восстановить данные для всех вкладок приложения."
        "\n\nОднако не всегда возможно восстановить данные для вкладки **Настройки моделей**. "
        "В таких случаях вкладки **Аналитика** и **Скважина** все еще будут работать."
    )
    uploaded_state = st.file_uploader(
        'Загрузить готовое состояние программы:',
        type='pickle',
        help="""Входной файл должен иметь расширение **.pickle**"""
    )
    if uploaded_state is not None:
        try:
            saved_state = pickle.load(uploaded_state)
            restore_saved_state(saved_state, state, session)
            st.success("Расчеты обработаны успешно! "
                       "Обновлены вкладки **Карта скважин**, **Аналитика** и **Скважина**.")
        except pickle.UnpicklingError as err:
            st.error('Не удалось восстановить расчеты.', err)
            logger.exception('Не удалось восстановить расчеты.', err)


def restore_saved_state(saved_state: Mapping[str, Any], state: AppState, session: Optional[Any] = None) -> None:
    for key, value in saved_state.items():
        state[key] = value

    from main_UI import apply_state_defaults, sync_legacy_session_state

    apply_state_defaults(state)
    repair_imported_state(state)

    if session is not None:
        sync_legacy_session_state(session, state)


def repair_imported_state(state: AppState) -> None:
    config = state.was_config
    if config is not None:
        if not state.field_name:
            state.field_name = config.field_name
        if not state.shops:
            state.shops = config.shops
        state.setdefault("was_date_start", config.date_start)
        state.setdefault("was_date_test", config.date_test)
        state.setdefault("was_date_end", config.date_end)

    if state.was_date_test is not None and state.was_date_test_if_ensemble is None:
        state.was_date_test_if_ensemble = state.was_date_test + dt.timedelta(days=state.ensemble_adapt_period)

    repair_imported_well_selection(state)
    repair_imported_statistics_slice(state)
    state.buffer = None


def repair_imported_well_selection(state: AppState) -> None:
    derived_wells = derive_wells_from_statistics(state.statistics or {})
    if not state.selected_wells_norm and derived_wells:
        state.selected_wells_norm = derived_wells

    if (not state.wellnames_key_normal or not state.wellnames_key_ois) and state.was_config is not None:
        from main_UI import load_well_mapping

        normal_to_ois, ois_to_normal = load_well_mapping(state.was_config.field_name, state.was_config.shops)
        if normal_to_ois and ois_to_normal:
            state.wellnames_key_normal = normal_to_ois
            state.wellnames_key_ois = ois_to_normal

    if not state.wellnames_key_normal and state.selected_wells_norm:
        state.wellnames_key_normal = {well: well for well in state.selected_wells_norm}
    if not state.wellnames_key_ois and state.selected_wells_norm:
        state.wellnames_key_ois = {well: well for well in state.selected_wells_norm}


def derive_wells_from_statistics(statistics: Mapping[str, pd.DataFrame]) -> list:
    wells = set()
    for df in statistics.values():
        for column in df.columns:
            parts = str(column).rsplit("_", 2)
            if len(parts) == 3 and parts[1] in {"liq", "oil", "gas"} and parts[2] in {"true", "pred"}:
                wells.add(parts[0])
    return sorted(wells)


def repair_imported_statistics_slice(state: AppState) -> None:
    has_statistics = bool(state.statistics)
    has_test_slice = bool(state.statistics_test_only)
    has_test_index = state.statistics_test_index is not None and len(state.statistics_test_index) > 0
    if not has_statistics or (has_test_slice and has_test_index):
        return

    from UI.data_processor import cut_statistics_test_only

    state.statistics_test_only, state.statistics_test_index = cut_statistics_test_only(state)


def draw_export_state(state: AppState) -> None:
    st.subheader("Экспорт текущего состояния программы")
    if state.was_config is None:
        st.info('Кнопка станет доступна, как только будет рассчитана хотя бы одна скважина.')
        return

    try:
        state_to_save = {key: state[key] for key in state}
        state_to_save = pickle.dumps(state_to_save)
        st.download_button(
            label="Экспорт текущего состояния программы",
            data=state_to_save,
            file_name=f"{state.was_config.field_name}_{state.was_date_test}_{state.was_date_end}.pickle",
            mime='application/octet-stream',
        )
    except pickle.PicklingError as err:
        text_error = 'Не удалось создать файл для экспорта. Результаты расчетов в формате .xlsx можно '
        'экспортировать по кнопке ниже.'
        st.error(text_error)
        logger.exception(text_error, err)


def draw_export_excel(state: AppState) -> None:
    st.subheader("Экспорт результатов по всем скважинам в Excel-формате (.xlsx)")
    st.write("""**Внимание!** Результаты, экспортированные в формате .xlsx, будет 
        невозможно импортировать как состояние программы.""")
    # Подготовка данных к выгрузке
    if state.statistics:
        if state.buffer is None:
            state.buffer = io.BytesIO()
            with pd.ExcelWriter(state.buffer) as writer:
                for key in state.statistics:
                    state.statistics[key].to_excel(writer, sheet_name=ConfigStatistics.MODEL_NAMES[key])
                if not state.ensemble_interval.empty:
                    state.ensemble_interval.to_excel(writer, sheet_name='Доверит. интервал ансамбль')
                if state.adapt_params:
                    df_adapt_params = pd.DataFrame(state.adapt_params)
                    df_adapt_params.to_excel(writer, sheet_name='Параметры адаптации пьезо')
                if state.models_weights:
                    for mode in state.models_weights.keys():
                        pd.DataFrame.from_dict(state.models_weights[mode]).to_excel(writer, sheet_name=f'Веса моделей {mode}')

        st.download_button(label="Экспорт .xlsx",
                           data=state.buffer,
                           file_name=f'Все результаты {state.was_config.field_name} {state.was_date_test}_{state.was_date_end}.xlsx',
                           mime='text/csv', )
    else:
        st.info("Кнопка станет доступна, как только будет рассчитана хотя бы одна скважина.")


def draw_upload_oilfield_data() -> None:
    st.subheader("Загрузка входных данных по месторождению")
    oilfield_name = st.text_input('Введите название месторождения', max_chars=30)
    oilfield_shops = st.text_input('Введите название цехов',
                                   max_chars=30,
                                   help="""Введите название цехов в данном месторождении через запятую. 
                                         Например 'ЦДНГ-4', 'ЦДНГ-2' (H - латинская)""")
    oilfield_files = st.file_uploader('Загрузить данные по месторождению:',
                                      accept_multiple_files=True,
                                      type=['feather', 'xlsm'])
    button_add_oilfield = st.button('OK')
    if button_add_oilfield:
        add_oilfield(oilfield_name, oilfield_files)


def add_oilfield(oilfield_name: str,
                 oilfield_files: IO) -> None:
    path_to_save = Path.cwd() / 'tools_preprocessor' / 'data' / oilfield_name
    if not path_to_save.exists():
        path_to_save.mkdir()
    for file in oilfield_files:
        with open(path_to_save / file.name, 'wb') as f:
            f.write(file.getbuffer())
    if Path(path_to_save / 'welllist.feather').exists():
        oilfield_shops = pd.read_feather(Path(path_to_save / 'welllist.feather'))
        FIELDS_SHOPS[oilfield_name] = list(oilfield_shops.ceh.unique())

def external_stats(state: AppState):
    st.subheader("Загрузка готовых данных для статистики и Ансамбля")
    uploaded_file = st.file_uploader('Принимаются данные в формате .xlsx',
                                    accept_multiple_files=False,
                                      type='xlsx')
    if uploaded_file is None:
        return

    state.statistics[uploaded_file.name.split('.')[0]] = pd.read_excel(uploaded_file, sheet_name=0, index_col=0)
    state.statistics_test_only[uploaded_file.name.split('.')[0]] = pd.read_excel(uploaded_file, sheet_name=0, index_col=0)
    state['statistics_another_models'] = uploaded_file.name.split('.')[0]
