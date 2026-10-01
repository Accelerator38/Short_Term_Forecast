import streamlit as st
import pandas as pd
import datetime
import json
import numpy
from copy import deepcopy
from UI.config import ML_FULL_ABBR, YES_NO, DEFAULT_FTOR_BOUNDS
from frameworks_shelf_algo.class_Shelf.config import ConfigShelf
from frameworks_shelf_algo.class_Shelf.support_functions import _get_path, \
    transform_str_dates_to_datetime_or_vice_versa, get_s_decline_rates, get_s_decline_rates_liq, get_s_decline_rates_gas
from frameworks_shelf_algo.class_Shelf.data_processor_shelf import DataProcessorShelf
from tools_preprocessor.config import Config as ConfigPreprocessor
from tools_preprocessor.preprocessor import Preprocessor
from UI.cached_funcs import run_preprocessor #, parse_well_names
# from UI.pages.tp_settings import draw_last_measurement_settings, draw_decline_rates_settings
from frameworks_shelf_algo.class_Shelf.constants import LAST_MEASUREMENT, DATE, \
    VALUE, VALUE_LIQ, VALUE_GAS, DEC_RATES, DEC_RATES_LIQ, DEC_RATES_GAS


def show(session: st.session_state) -> None:
    draw_ftor_settings(session)
    draw_wolfram_settings(session)
    draw_CRM_settings(session)
    draw_shelf_settings(session)
    draw_ensemble_settings(session)


def draw_ftor_settings(session: st.session_state) -> None:
    with st.expander('Настройки модели пьезопроводности'):
        with st.form(key='ftor_bounds'):
            for param_name, param_dict in DEFAULT_FTOR_BOUNDS.items():
                cols = st.columns([0.4, 0.2, 0.2, 0.2])
                cols[0].checkbox(
                    label=param_dict['label'],
                    value=session[f'{param_name}_is_adapt'],
                    key=f'{param_name}_is_adapt_',
                    help=param_dict['help']
                )
                cols[1].number_input(
                    label='От',
                    min_value=param_dict['min'],
                    value=session[f'{param_name}_lower'],
                    max_value=param_dict['max'],
                    step=param_dict['step'],
                    key=f'{param_name}_lower_'
                )
                cols[2].number_input(
                    label='Фиксированное',
                    min_value=param_dict['min'],
                    value=session[f'{param_name}_default'],
                    max_value=param_dict['max'],
                    step=param_dict['step'],
                    key=f'{param_name}_default_'
                )
                cols[3].number_input(
                    label='До',
                    min_value=param_dict['min'],
                    value=session[f'{param_name}_upper'],
                    max_value=param_dict['max'],
                    step=param_dict['step'],
                    key=f'{param_name}_upper_',
                    help='включительно'
                )
            submit_bounds = st.form_submit_button('Применить',
                                                  on_click=update_ftor_constraints,
                                                  kwargs={'write_from': session,
                                                          'write_to': session})
        button_use_GDIS = st.button('Использовать границы из ГДИС')
        if button_use_GDIS:
            session.constraints = {}


def draw_wolfram_settings(session: st.session_state) -> None:
    with st.expander('Настройки модели ML'):
        with st.form(key='ML_params'):
            st.selectbox(
                label='Модель на 1-ом уровне',
                options=ML_FULL_ABBR.keys(),
                index=2,  # XGBoost
                help="""
                    Данная модель использует для обучения только входные данные.
                    Подробнее о моделях см. [sklearn](https://scikit-learn.org) 
                    и [xgboost](https://xgboost.readthedocs.io).
                    """,
                key='estimator_name_group_'
            )
            st.selectbox(
                label='Модель на 2-ом уровне',
                options=ML_FULL_ABBR.keys(),
                index=1,  # LinearSVR
                help="""
                    Данная модель использует для обучения как входные данные, 
                    так и результаты работы модели 1-ого уровня.
                    Подробнее о моделях см. [sklearn](https://scikit-learn.org) 
                    и [xgboost](https://xgboost.readthedocs.io).
                    """,
                key='estimator_name_well_'
            )
            st.selectbox(
                label='Cross Validation на 2-ом уровне',
                options=YES_NO.keys(),
                index=1,  # Нет
                help="""
                    Данная процедура нацелена на предотвращение переобучения модели 2-ого уровня.
                    Подробнее см. [Cross-validation for time series](https://robjhyndman.com/hyndsight/tscv).
                    """,
                key='is_deep_grid_search_'
            )
            st.text_input(
                label='Размеры скользящего окна',
                value='3 5 7 15 30',
                max_chars=20,
                help="""Укажите через пробел""",
                key='window_sizes_'
            )
            st.text_input(
                label='Квантили',
                value='0.1 0.3',
                max_chars=20,
                help="""Укажите через пробел""",
                key='quantiles_'
            )

            submit_params = st.form_submit_button('Применить',
                                                  on_click=update_ML_params,
                                                  kwargs={'write_from': session,
                                                          'write_to': session})


def draw_CRM_settings(session: st.session_state) -> None:
    with st.expander('Настройки модели CRM'):
        with st.form(key='CRM_params'):
            st.number_input(
                label='Радиус влияния, м',
                min_value=300,
                value=session.CRM_influence_R,
                max_value=10000,
                step=100,
                help="""Вне радиуса скважины считаются невзаимодействующими""",
                key='CRM_influence_R_'
            )
            st.number_input(
                label='Оптимизатор: максимальное число итераций',
                min_value=1,
                value=session.CRM_maxiter,
                max_value=1000,
                step=50,
                key='CRM_maxiter_'
            )
            st.number_input(
                label='Пластовое давление',
                min_value=100,
                value=session.CRM_p_res,
                max_value=1000,
                step=50,
                help="""Параметр для функции восстановления давления""",
                key='CRM_p_res_'
            )

            submit_params = st.form_submit_button('Применить',
                                                  on_click=update_CRM_params,
                                                  kwargs={'write_from': session,
                                                          'write_to': session})


def draw_shelf_settings(session: st.session_state) -> None:
    with st.expander('Настройки модели ППТП'):
        # сопоставляет имена выбранных скважин OIS и (ГРАД?)
        #     wellnames_key_normal : Dict[str, int]
        #         Ключ = имя скважины в формате ГРАД, значение - имя скважины OIS.
        #     wellnames_key_ois : Dict[int, str]
        #         Ключ = имя скважины OIS, значение - имя скважины в формате ГРАД.
        if 'change_gtm_info' not in session:
            session['change_gtm_info'] = 0
        if 'change_gtm' not in session:
            session['change_gtm'] = 0
        if 'change_tp' not in session:
            session['change_tp'] = 0
        # if session['changes'] == True:
        #     session['change_gtm_info'] = 0
        session['changes'] = False
        file = st.file_uploader('Загрузить данные', type='json')
        _path = _get_path(session.field_name)
        # if session['change_gtm_info'] == 0:
        if session['change_gtm'] == 0:
            welllist = pd.read_feather(_path / 'welllist.feather')
            wells_work = pd.read_feather(_path / 'sh_sost_fond.feather')
            wells_work.set_index('dt', inplace=True)
            wells_work = wells_work[wells_work.index > session.date_test]
            wells_work = wells_work[wells_work["sost"] == 'В работе']
            wells_work = wells_work[wells_work["charwork.name"] == 'Нефтяные']
            all_wells_ois_ = wells_work["well.ois"]
            wellnames_key_normal_ = {}
            wellnames_key_ois_ = {}
            for ois_well in all_wells_ois_.unique():
                well_name_norm = welllist[welllist["ois"] == ois_well]
                well_name_norm = well_name_norm[well_name_norm.npath == 0]
                well_name_norm = well_name_norm.at[well_name_norm.index[0], 'num']
                wellnames_key_normal_[well_name_norm] = ois_well
                wellnames_key_ois_[ois_well] = well_name_norm
        else:
            wellnames_key_normal_ = session.state.wellnames_key_normal
            wellnames_key_ois_ = session.state.wellnames_key_ois
        if file is not None:
            data = json.load(file)
            transform_str_dates_to_datetime_or_vice_versa(data, dates_to_datetime=True)
            new_keys = []
            for w in data.keys():
                if w != 'MLSP_STOP':
                    new_keys.append(numpy.int64(w))
                else:
                    new_keys.append(w)
            data = dict(zip(new_keys, list(data.values())))
            session.shelf_json = data
            session['change_gtm_info'] = session['change_gtm_info'] + 1
        else:
            with st.form(key='shelf_params'):
                max_adapt_period = (session.date_test - session.date_start).days - 1
                # if max_adapt_period <= 25:
                #     max_adapt_period = 30
                st.number_input(
                    label='Количество дней для расчета темпа падения',
                    min_value=2,
                    value=session.n_days_past,
                    max_value=max_adapt_period,
                    step=1,
                    key='n_days_past_'
                )
                max_avg = int(session.n_days_past / 2)
                st.number_input(
                    label='Количество дней для осреднения',
                    min_value=1,
                    value=session.n_days_calc_avg,
                    max_value=max_avg,
                    step=1,
                    key='n_days_calc_avg_'
                )
                submit_params = st.form_submit_button('Применить',
                                                      on_click=update_shelf_params,
                                                      kwargs={'write_from': session,
                                                              'write_to': session})
            if session.selected_wells_norm:
                if 'Все скважины' in session.selected_wells_norm:
                    wells_ois = list(wellnames_key_ois_.keys())
                else:
                    wells_ois = [wellnames_key_normal_[well_name_] for well_name_ in session.selected_wells_norm]
                wells_sorted_ois = sorted(wells_ois)
                config_shelf = ConfigShelf(oilfield=session.field_name,
                                           shops=session.shops,
                                           wells_ois=wells_sorted_ois,
                                           train_start=session.date_start,
                                           train_end=session.date_test,
                                           predict_start=session.date_test,
                                           predict_end=session.date_end,
                                           n_days_past=session.n_days_past,
                                           n_days_calc_avg=session.n_days_calc_avg)
                # if 'change_gtm' not in session:
                #     session['change_gtm'] = 0
                # if session['change_gtm_info'] == 0:
                DataProcessorShelf(config_shelf)
                session['change_gtm_info'] = session['change_gtm_info'] + 1
        st.write('-' * 100)
        st.write('**Последний замер и темпы падения**')
        if session.selected_wells_norm:
            # if 'Все скважины' in session.selected_wells_norm:
            #     wells_ois = list(wellnames_key_ois_.keys())
            # else:
            #     wells_ois = [wellnames_key_normal_[well_name_] for well_name_ in session.selected_wells_norm]
            # wells_sorted_ois = sorted(wells_ois)
            # config_shelf = ConfigShelf(oilfield=session.field_name,
            #                            shops=session.shops,
            #                            wells_ois=wells_sorted_ois,
            #                            train_start=session.date_start,
            #                            train_end=session.date_test,
            #                            predict_start=session.date_test,
            #                            predict_end=session.date_end,
            #                            n_days_past=session.n_days_past,
            #                            n_days_calc_avg=session.n_days_calc_avg)
            # # if 'change_gtm' not in session:
            # #     session['change_gtm'] = 0
            # # if session['change_gtm_info'] == 0:
            # DataProcessorShelf(config_shelf)
            # session['change_gtm_info'] = session['change_gtm_info'] + 1
            if 'Все скважины' in session.selected_wells_norm:
                wells_ois = list(session.shelf_json.keys())
                del wells_ois[0]
            wells_sorted_ois = sorted(wells_ois)
            # for w in wells_sorted_ois:
            #     print(w, type(w))
            # print("==========")
            # print(wellnames_key_ois_)
            # for wo in wellnames_key_ois_.keys():
            #     print(wo, type(wo))

            wells_sorted_norm = [wellnames_key_ois_[int(w)] for w in wells_sorted_ois]
            # wells_sorted_norm = [wellnames_key_ois_[w] for w in wells_sorted_ois]
            _well1 = st.selectbox(
                label='Скважина',
                options=wells_sorted_norm,
                key='well',
            )
            _well = wellnames_key_normal_[_well1]
            _date_start = session['date_test']
            draw_last_measurement_settings(_well, _date_start)
            st.write('-' * 100)
            _date_end = session['date_end']
            draw_decline_rates_settings(_well, _date_start, _date_end)
            data_out = deepcopy(session.shelf_json)
            new_keys = []
            for w in data_out.keys():
                if w != 'MLSP_STOP':
                    new_keys.append(int(w))
                else:
                    new_keys.append(w)
            data_out = dict(zip(new_keys, list(data_out.values())))
            transform_str_dates_to_datetime_or_vice_versa(data_out, dates_to_datetime=False)
            st.download_button('Выгрузить данные', json.dumps(data_out, ensure_ascii=False), 'data_output.json')
        else:
            st.write("Необходимо выбрать скважину")


def draw_ensemble_settings(session: st.session_state) -> None:
    with st.expander('Настройки модели ансамбля'):
        with st.form(key='ensemble_params'):
            max_adapt_period = (session.date_test - session.date_start).days - 1
            if max_adapt_period <= 25:
                max_adapt_period = 30
            st.number_input(
                label='Количество дней обучения ансамбля',
                min_value=25,
                value=session.ensemble_adapt_period,
                max_value=max_adapt_period,
                step=1,
                key='ensemble_adapt_period_'
            )
            st.number_input(
                label='Значимость доверительного интервала (от 0 до 1)',
                min_value=0.01,
                value=session.interval_probability,
                max_value=1.,
                step=0.01,
                key='interval_probability_'
            )
            st.number_input(
                label='Draws',
                min_value=100,
                value=session.draws,
                max_value=10000,
                step=10,
                help="""The number of samples to draw. The number of tuned samples are discarded by default.""",
                key='draws_'
            )
            st.number_input(
                label='Tune',
                min_value=100,
                value=session.tune,
                max_value=1000,
                step=10,
                help="""Number of iterations to tune, defaults to 1000. Samplers adjust the step sizes, scalings or
                        similar during tuning. Tuning samples will be drawn in addition to the number specified in
                        the ``draws`` argument.""",
                key='tune_'
            )
            st.number_input(
                label='Chains',
                min_value=1,
                value=session.chains,
                max_value=5,
                step=1,
                help="""The number of chains to sample. Running independent chains is important for some
                        convergence statistics and can also reveal multiple modes in the posterior.""",
                key='chains_'
            )
            st.number_input(
                label='Target_accept',
                min_value=0.01,
                value=session.target_accept,
                max_value=1.,
                step=0.01,
                help="""The step size is tuned such that we approximate this acceptance rate. 
                        Higher values like 0.9 or 0.95 often work better for problematic posteriors""",
                key='target_accept_'
            )
            submit_params = st.form_submit_button('Применить',
                                                  on_click=update_ensemble_params,
                                                  kwargs={'write_from': session,
                                                          'write_to': session})


def update_ftor_constraints(write_from: st.session_state,
                            write_to: st.session_state) -> None:
    # TODO: костыль для многостраничности: приходится записывать параметры модели в session и подтягивать
    #  их для каждой следующей отрисовки. Изменить, когда выйдет версия Streamlit multipage. (~1 квартал 2022)
    for param_name, param_dict in DEFAULT_FTOR_BOUNDS.items():
        write_to[f'{param_name}_is_adapt'] = write_from[f'{param_name}_is_adapt_']
        write_to[f'{param_name}_lower'] = write_from[f'{param_name}_lower_']
        write_to[f'{param_name}_default'] = write_from[f'{param_name}_default_']
        write_to[f'{param_name}_upper'] = write_from[f'{param_name}_upper_']

    constraints = {}
    for param_name, param_dict in DEFAULT_FTOR_BOUNDS.items():
        # Если параметр нужно адаптировать
        if write_to[f'{param_name}_is_adapt']:
            constraints[param_name] = [write_to[f'{param_name}_lower'], write_to[f'{param_name}_upper']]
        else:
            # Если значение параметра нужно зафиксировать
            constraints[param_name] = write_to[f'{param_name}_default']
    write_to.constraints = constraints


def update_ML_params(write_from: st.session_state,
                     write_to: st.session_state) -> None:
    write_to['estimator_name_group'] = ML_FULL_ABBR[write_from['estimator_name_group_']]
    write_to['estimator_name_well'] = ML_FULL_ABBR[write_from['estimator_name_well_']]
    write_to['is_deep_grid_search'] = YES_NO[write_from['is_deep_grid_search_']]
    write_to['window_sizes'] = [int(ws) for ws in write_from['window_sizes_'].split()]
    write_to['quantiles'] = [float(q) for q in write_from['quantiles_'].split()]


def update_CRM_params(write_from: st.session_state,
                      write_to: st.session_state) -> None:
    write_to['CRM_influence_R'] = int(write_from['CRM_influence_R_'])
    write_to['CRM_maxiter'] = int(write_from['CRM_maxiter_'])
    write_to['CRM_p_res'] = int(write_from['CRM_p_res_'])


def update_shelf_params(write_from: st.session_state,
                        write_to: st.session_state) -> None:
    write_to['n_days_past'] = int(write_from['n_days_past_'])
    write_to['n_days_calc_avg'] = int(write_from['n_days_calc_avg_'])
    # st.session_state['change_tp'] = 0
    st.session_state['change_gtm_info'] = 0


def update_ensemble_params(write_from: st.session_state,
                           write_to: st.session_state) -> None:
    write_to['interval_probability'] = write_from['interval_probability_']
    write_to['draws'] = int(write_from['draws_'])
    write_to['tune'] = int(write_from['tune_'])
    write_to['chains'] = int(write_from['chains_'])
    write_to['target_accept'] = float(write_from['target_accept_'])
    write_to['ensemble_adapt_period'] = int(write_from['ensemble_adapt_period_'])


def draw_last_measurement_settings(_well: str, _date_start: datetime.date):
    def edit_last_measurement():
        st.session_state.shelf_json[_well][LAST_MEASUREMENT][DATE] = st.session_state['changed_date']
        st.session_state.shelf_json[_well][LAST_MEASUREMENT][VALUE] = st.session_state['changed_val']
        st.session_state.shelf_json[_well][LAST_MEASUREMENT][VALUE_LIQ] = st.session_state['changed_val_liq']
        st.session_state.shelf_json[_well][LAST_MEASUREMENT][VALUE_GAS] = st.session_state['changed_val_gas']
        st.session_state['change_gtm_info'] = st.session_state['change_gtm_info'] + 1

    def del_last_measurement():
        st.session_state.shelf_json[_well][LAST_MEASUREMENT] = dict()
        st.session_state['change_gtm_info'] = st.session_state['change_gtm_info'] + 1

    st.write('**Последний замер**')
    last_measurement_data = st.session_state.shelf_json[_well][LAST_MEASUREMENT]
    there_is_data = len(last_measurement_data) != 0
    if there_is_data:
        date = last_measurement_data[DATE]
        val = last_measurement_data[VALUE]
        val_liq = last_measurement_data[VALUE_LIQ]
        val_gas = last_measurement_data.get(VALUE_GAS, 0.0)
        st.write(f"Дата: {date}")
        st.write(f"Значение нефти: {val}")
        st.write(f"Значение жидкости: {val_liq}")
        st.write(f"Значение газа: {val_gas}")
    else:
        st.write('Данные отсутствуют')
        date, val, val_liq, val_gas = _date_start, 0, 0, 0
    with st.empty():
        if 'b_edit_last_measurement' in st.session_state and st.session_state['b_edit_last_measurement'] is True:
            with st.form('form_edit_last_measurement'):
                st.date_input('Дата', value=date, key='changed_date')
                st.number_input('Значение нефти', value=val, key='changed_val')
                st.number_input('Значение жидкости', value=val_liq, key='changed_val_liq')
                st.number_input('Значение газа', value=val_gas, key='changed_val_gas')
                st.form_submit_button('Применить', on_click=edit_last_measurement)
        else:
            st.button('Добавить/изменить', key='b_edit_last_measurement')
    if there_is_data:
        st.button('Удалить', on_click=del_last_measurement)

def draw_decline_rates_settings(_well: str, _date_start: datetime.date, _date_end: datetime.date):
    def edit_dec_rate():
        st.session_state.shelf_json[_well][DEC_RATES][st.session_state['new_date']] = st.session_state['new_val']
        st.session_state.shelf_json[_well][DEC_RATES_LIQ][st.session_state['new_date']] = st.session_state['new_val_liq']
        st.session_state.shelf_json[_well][DEC_RATES_GAS][st.session_state['new_date']] = st.session_state['new_val_gas']
        st.session_state['change_gtm_info'] = st.session_state['change_gtm_info'] + 1
        # print("edit TP")
        # print(_well)

    def del_dec_rate():
        del st.session_state.shelf_json[_well][DEC_RATES][st.session_state['date_to_delete']]
        del st.session_state.shelf_json[_well][DEC_RATES_LIQ][st.session_state['date_to_delete']]
        del st.session_state.shelf_json[_well][DEC_RATES_GAS][st.session_state['date_to_delete']]
        st.session_state['change_gtm_info'] = st.session_state['change_gtm_info'] + 1

    st.write('**Темпы падения**')
    st.session_state.shelf_json[_well].setdefault(DEC_RATES_GAS, {})
    col1, col2 = st.columns(2)
    with col1:
        st.write('Введенные пользователем:')
        dec_rates_to_show = deepcopy(st.session_state.shelf_json[_well][DEC_RATES])
        dec_rates_to_show_liq = deepcopy(st.session_state.shelf_json[_well][DEC_RATES_LIQ])
        dec_rates_to_show_gas = deepcopy(st.session_state.shelf_json[_well][DEC_RATES_GAS])
        transform_str_dates_to_datetime_or_vice_versa(dec_rates_to_show, dates_to_datetime=False)
        transform_str_dates_to_datetime_or_vice_versa(dec_rates_to_show_liq, dates_to_datetime=False)
        transform_str_dates_to_datetime_or_vice_versa(dec_rates_to_show_gas, dates_to_datetime=False)
        st.write('Темпы падения для нефти')
        st.write(dec_rates_to_show)
        st.write('Темпы падения для жидкости')
        st.write(dec_rates_to_show_liq)
        st.write('Темпы падения для газа')
        st.write(dec_rates_to_show_gas)
        with st.empty():
            if 'b_edit_dec_rate' in st.session_state and st.session_state['b_edit_dec_rate'] is True:
                with st.form('form_edit_dec_rate'):
                    st.date_input('Дата', value=_date_start, key='new_date')
                    st.number_input('Значение ТП нефти', key='new_val')
                    st.number_input('Значение ТП жидкости', key='new_val_liq')
                    st.number_input('Значение ТП газа', key='new_val_gas')
                    st.form_submit_button('Добавить/изменить', on_click=edit_dec_rate)
            else:
                st.button('Добавить/изменить', key='b_edit_dec_rate')
        with st.empty():
            if 'b_del_dec_rate' in st.session_state and st.session_state['b_del_dec_rate'] is True:
                with st.form('form_del_dec_rate'):
                    dates_when_dec_rate_changes = [*st.session_state.shelf_json[_well][DEC_RATES].keys()]
                    st.selectbox('Дата', dates_when_dec_rate_changes, key='date_to_delete')
                    st.form_submit_button('Удалить', on_click=del_dec_rate)
            else:
                if len(st.session_state.shelf_json[_well][DEC_RATES]) != 0:
                    st.button('Удалить', key='b_del_dec_rate')
    with col2:
        st.write('Автозаполненные:')
        s_decline_rates = get_s_decline_rates(st.session_state.shelf_json, _well, _date_start, _date_end)
        s_decline_rates_liq = get_s_decline_rates_liq(st.session_state.shelf_json, _well, _date_start, _date_end)
        s_decline_rates_gas = get_s_decline_rates_gas(st.session_state.shelf_json, _well, _date_start, _date_end)
        pd_decline_show = pd.concat([s_decline_rates, s_decline_rates_liq, s_decline_rates_gas], axis=1, ignore_index=True)
        pd_decline_show.columns = ['нефть', 'жидкость', 'газ']
        st.write(pd_decline_show)
