import os
import logging.config
import shelve
import config

from datetime import datetime
from datetime import timedelta

import pandas as pd
import numpy as np


def recounted_inj(inj_wells_f, inj_names):
    '''
    Пересчитывает приемистость умножая её на коэффициент взаимовлияния
    :param inj_wells_f: коэффициент взаимовлияния суммарный для скважины
    :param inj_names: имена нагнетающих скважин
    :return: DateFrame с пересчитанной приемистостью
    '''
    # TODO rename
    inj_recounted_dict = {}
    # TODO переделать
    for i, well_name in enumerate(inj_names):
        inj_recounted_dict[well_name] = inj_wells_f[i] * injection[well_name]
    inj_recounted_df = pd.DataFrame(inj_recounted_dict)

    return inj_recounted_df

def get_info(injection, pressure, ml_wells):
    info = {
        'name': 'hybrid',
        'version': '0.0.1',
        'restart_date': datetime.strftime(injection.index[-1] + timedelta(1), '%Y-%m-%d'),
        'prod_wells': {},
        'inj_wells': {}
    }
    for (well_name, inj) in injection.iteritems():
        info['inj_wells'][well_name] = {'injection': inj[-1]}

    for (well_name, press) in pressure.iteritems():
        info['prod_wells'][well_name] = {'pressure': press[-1]}

    return info



def get_synonyms(coordinates):
    coordinates.sort_values(by=list(coordinates.columns[-2:]), inplace=True)
    synonyms_pairs = []
    prev = coordinates.iloc[0]
    for ind, cur in coordinates.iloc[1:].iterrows():
        if (cur.iloc[0] == prev.iloc[0]) and (cur.iloc[1] == prev.iloc[1]):
            synonyms_pairs.append((prev.name, cur.name))
        prev = cur

    synonyms_dict = {}
    if len(synonyms_pairs) > 0:
        if len(synonyms_pairs) > 1:
            prev = synonyms_pairs[0]
            ind = 1
            tmp_lst = [prev[0], prev[1]]
            while True:
                if ind >= len(synonyms_pairs):
                    break

                cur = synonyms_pairs[ind]
                if prev[1] == cur[0]:
                    tmp_lst.append(cur[1])
                else:
                    true_name = min(tmp_lst, key=len)
                    synonyms_dict[true_name] = tmp_lst

                    tmp_lst = [cur[0], cur[1]]

                prev = synonyms_pairs[ind]
                ind += 1

            true_name = min(tmp_lst, key=len)
            synonyms_dict[true_name] = tmp_lst
        else:
            true_name = min(synonyms_pairs[0], key=len)
            synonyms_dict[true_name] = [el for el in synonyms_pairs[0]]

    return synonyms_dict


def change_cut_dates_nes(ind_left, ind_right):
    '''
    Меняет даты если, не входят в CONFIG
    '''
    global config
    print(ind_left, ind_right)
    if datetime.strptime(config.CUT_DATE, '%Y-%m-%d') < ind_left:
        config.CUT_DATE = datetime.strftime(ind_left, '%Y-%m-%d')
        logger.info('CUT_DATE changed')
        print('CUT_DATE changed')

    if datetime.strptime(config.CUT_DATE2, '%Y-%m-%d') > ind_right:
        config.CUT_DATE2 = datetime.strftime(ind_right, '%Y-%m-%d')
        logger.info('CUT_DATE2 changed')
        print('CUT_DATE2 changed')


log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf')
logging.config.fileConfig(log_file_path)
logger = logging.getLogger('crm.__main__')

coordinates = None
production = None
pressure = None
pressure_reservoir = None
injection = None
water_cut = None
gasfact = None


def main():
    import json
    import pickle
    import prepare_feather_data
    from functools import reduce

    from matplotlib import pyplot as plt
    #
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    #
    from crm.preprocessing.utils import cut_inactive_wells
    from crm.preprocessing.utils import select_wells_nearest_to_prod
    from crm.preprocessing.utils import recover_pres
    from crm.models.crm import CRM
    from crm.models.crmip import CRMIP
    from crm.optimization.minimizers import SciPySlsqpWrapper

    from hybrid import Base_TSF_Model as TimeSeriesModel
    from hybrid import prepare_injection as pdInj
    from hybrid import utils

    from tpot import TPOTRegressor

    import plot


    global coordinates, production, pressure, pressure_reservoir, injection, water_cut, gasfact

    logger.info('preset_choice ' + str(config.preset_choice))
    logger.info('path_train ' + str(config.path_train))
    logger.info('models_save_path_train ' + str(config.models_save_path_train))
    logger.info('save_path_output_pipeline_train ' + str(config.save_path_output_pipeline_train))
    logger.info('INFLUENCE_R ' + str(config.INFLUENCE_R))
    logger.info('double_R ' + str(config.double_R))
    logger.info('split_ratio ' + str(config.split_ratio))
    logger.info('inactive_time ' + str(config.inactive_time))
    logger.info('CUT_DATE ' + str(config.CUT_DATE))
    logger.info('CUT_DATE2 ' + str(config.CUT_DATE2))
    logger.info('p_res ' + str(config.p_res))
    logger.info('p_0 ' + str(config.p_0))
    logger.info('recov_time ' + str(config.recov_time))
    logger.info('TR_DAY ' + str(config.TR_DAY))
    logger.info('TAU_0 ' + str(config.TAU_0))
    logger.info('J_0 ' + str(config.J_0))
    logger.info('generations ' + str(config.generations))
    logger.info('population_size ' + str(config.population_size))
    logger.info('early_stop ' + str(config.early_stop))
    logger.info('options_SLSQP_CRM ' + str(config.options_SLSQP_CRM))
    logger.info('liquid_flag ' + str(config.liquid_flag))
    logger.info('wc_flag ' + str(config.wc_flag))
    logger.info('gas_factor_flag ' + str(config.gas_factor_flag))
    logger.info('pressure_reservoir_flag ' + str(config.pressure_reservoir_flag))
    logger.info('prod_bnd ' + str(config.prod_bnd))
    logger.info('tau_bnd ' + str(config.tau_bnd))
    logger.info('J_bnd ' + str(config.J_bnd))
    logger.info('f_bnd ' + str(config.f_bnd))

    config.save_path_train = os.path.join(config.save_path_output_pipeline_train, 'CRM_output')
    config.bnds_CRM = {'prod_bnd': config.prod_bnd, 'tau_bnd': config.tau_bnd,
                       'J_bnd': config.J_bnd, 'f_bnd': config.f_bnd}

    save_path_output_pipeline_train_valid = os.path.join(config.save_path_output_pipeline_train, 'train_validation')
    config.test_period_len = ((1 - config.split_ratio) * (datetime.strptime(config.CUT_DATE2, '%Y-%m-%d') -
                                            datetime.strptime(config.CUT_DATE, '%Y-%m-%d'))).days - 1
    # Prepare data to CRM
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    prev_time = datetime.now()

    print('Time mark', datetime.now() - prev_time)
    xls = pd.ExcelFile(os.path.join(config.path_train, f'Data_model_reservoir_{config.oilfield}.xlsx')) # данные с месторождения
    if config.double_R == False:
        coordinates = pd.read_excel(xls, sheet_name=2, dtype={'Скважина': str})
        coordinates = coordinates[['Скважина', 'Координата X', 'Координата Y']]
    else:
        coordinates = pd.read_excel(xls, sheet_name=2, dtype={'Скважина': str})
    coordinates['Скважина'] = coordinates['Скважина'].astype(str)
    coordinates = coordinates.set_index('Скважина')
    print('Time mark', datetime.now() - prev_time)

    nef_info = pd.read_excel(xls, sheet_name=0, dtype={'Скважина': str})
    print('Time mark', datetime.now() - prev_time)
    production = pd.\
        pivot_table(nef_info, values='Дебит жидкости', index='Дата', columns='Скважина', dropna=False).\
        apply(lambda x: x.interpolate(limit_direction='forward', limit_area='inside'), axis=0)[config.CUT_DATE:config.CUT_DATE2].\
        dropna(axis=1, how='all').fillna(0) # заполняет пропуски дат в данных дебитах жидкости
    production.columns.name = None

    print('Time mark', datetime.now() - prev_time)
    water_cut = pd.\
        pivot_table(nef_info, values='Обводненность', index='Дата', columns='Скважина', dropna=False).\
        apply(lambda x: x.interpolate(limit_direction='forward', limit_area='inside'), axis=0)[config.CUT_DATE:config.CUT_DATE2].\
        dropna(axis=1, how='all') # заполняет пропуски дат в данных обводненности
    water_cut.columns.name = None

    print('Time mark', datetime.now() - prev_time)
    pressure = pd.\
        pivot_table(nef_info, values='Забойное давление', index='Дата', columns='Скважина', dropna=False).\
        apply(lambda x: x.interpolate(limit_direction='forward', limit_area='inside'), axis=0)[config.CUT_DATE:config.CUT_DATE2]. \
        dropna(axis=1, how='all').fillna(0) # заполняет пропуски дат в данных забойного давления
    pressure.columns.name = None

    print('Time mark', datetime.now() - prev_time)
    print('_______')

    nag_info = pd.read_excel(xls, sheet_name=1, dtype={'Скважина': str})
    print('Time mark', datetime.now() - prev_time)
    injection = pd.\
        pivot_table(nag_info, values='Приемистость', index='Дата', columns='Скважина', dropna=False).\
        apply(lambda x: x.interpolate(limit_direction='forward', limit_area='inside'), axis=0)[config.CUT_DATE:config.CUT_DATE2]. \
        dropna(axis=1, how='all').fillna(0) # заполняет пропуски дат в данных приемистости
    injection.columns.name = None

    print('Time mark', datetime.now() - prev_time)

    # индекс в datetime переводит
    production.index = pd.to_datetime(production.index)
    pressure.index = pd.to_datetime(pressure.index)
    water_cut.index = pd.to_datetime(water_cut.index)
    injection.index = pd.to_datetime(injection.index)

    # меняет граничные даты, если в config указаны даты не входящие в данные
    list([change_cut_dates_nes(el.index[0], el.index[-1]) for el in (production, water_cut, pressure, injection)])
    production = production.loc[config.CUT_DATE:config.CUT_DATE2]
    water_cut = water_cut.loc[config.CUT_DATE:config.CUT_DATE2]
    pressure = pressure.loc[config.CUT_DATE:config.CUT_DATE2]
    injection = injection.loc[config.CUT_DATE:config.CUT_DATE2]

    # делает переиндексацию на новые даты. Кажется, что можно без этой процедуры
    ind = pd.date_range(start=production.index[0], end=production.index[-1], freq='D')
    production = production.reindex(ind).\
        apply(lambda x: x.interpolate(method='linear', limit_direction='forward', limit_area='inside'), axis=0)

    pressure = pressure.reindex(ind).\
        apply(lambda x: x.interpolate(method='linear', limit_direction='forward', limit_area='inside'), axis=0)

    water_cut = water_cut.reindex(ind).\
        apply(lambda x: x.interpolate(method='linear', limit_direction='forward', limit_area='inside'), axis=0)

    injection = injection.reindex(ind).\
        apply(lambda x: x.interpolate(method='linear', limit_direction='forward', limit_area='inside'), axis=0)

    name_inj_wells = list(injection.columns)

    prod_df_lst = [production, pressure, water_cut]
    names_prod_wells = list(reduce(lambda x, y: x & set(y.columns), prod_df_lst[1:], set(prod_df_lst[0].columns)) # названия добывающих скважин
                            - set(name_inj_wells))

    for i, df in enumerate(prod_df_lst):
        prod_df_lst[i] = df[names_prod_wells]

        # в names_prod_wells нет и нагнетающих скважин,
        # и скважин, которых нет хотя бы в одной их характеристик добывающих скважин.
        # однако, непонятно, как информация о переводах вносится в приемистость.
        # поэтому необходимо занулить приемистость "из прошлого".
        # НЕ НУЖНО, ПОТОМУ ЧТО СЕЙЧАС наны в приемистости заполняются нулями и так. код с ошибками.
        # injection[(set(df.columns) - set(names_prod_wells)) & set(name_inj_wells)] = 0

    production, pressure, water_cut = prod_df_lst

    print(production.shape)
    print(pressure.shape)
    print(water_cut.shape)
    print(injection.shape)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # read_options = dict(index_col=0, parse_dates=True)
    path = config.path_train

    coordinates = coordinates[~coordinates.index.duplicated(keep='first')]

    # Матрицы, когда для дебита или приемистости есть соответствующее значение забойного давления (давления в линии)
    prod = set(injection.columns.values)

    # Скважины, которые простаивают большую часть времени необходимо исключить,
    # так как они вносят лишнюю неопределенность.
    active_prod_wells = cut_inactive_wells(production, thres=config.inactive_time)[0] # активные скважины
    production = production.loc[:, active_prod_wells.columns]
    pressure = pressure.loc[:, active_prod_wells.columns]
    water_cut = water_cut.loc[:, active_prod_wells.columns]

    prod = set(production.columns.values)
    pres = set(pressure.columns.values)
    list1 = list(pres & prod)
    production = production.loc[:, list1]
    pressure = pressure.loc[:, list1]
    # print(len(production.columns), len(pressure.columns))

    # print(len(production.columns), len(pressure.columns))

    # Нагнетательные скважины, удаленные на большое расстояние от добывающих вносят незначительный вклад либо вовсе
    # незначимы, поэтому для уменьшения размерности задачи их необходимо отбросить.
    names_prod_wells = list(set(production.columns) & set(coordinates.index))
    production = production[names_prod_wells]
    pressure = pressure[names_prod_wells]
    water_cut = water_cut[names_prod_wells]

    names_inj_wells = list(set(injection.columns) & set(coordinates.index))
    injection = injection[names_inj_wells]
    injection = select_wells_nearest_to_prod(injection, production, coordinates, dist_thres=config.INFLUENCE_R, double_R=config.double_R)

    # Для более точной аппроксимации значений дебитов при остановке и включении скважин, необходимо заполнить нулевые
    # значения для давлений. Для этого нужно знать значение пластового давления и
    # характерное время восстановления давления.
    # d1 =set(production.columns.values).difference(set(pressure.columns.values))
    # # print(set(production.columns.values))
    # # print(pressure.info())
    # # print(production.info())
    rec_pressure = recover_pres(pressure, inplace=False, p_res=config.p_res, p_0=config.p_0,
                                recov_time=config.recov_time) # заполняет значения в забойном давлении

    is_work = production.astype(bool).astype(int)
    is_halt = np.array(np.ones_like(is_work) - is_work, dtype=np.int64)

    # # Оптимизация модели
    # Для моделей CRM/CRMIP
    # схема модели svir/lwir
    SCHEME = config.SCHEME
    # начальное значение параметров модели
    TAU_0 = config.TAU_0
    J_0 = config.J_0
    # радиус влияния (вне его скважины считаются невзаимодействующими)
    INFLUENCE_R = config.INFLUENCE_R
    double_R = config.double_R

    # Параметры оптимизации SLSQP
    options = config.options_SLSQP_CRM

    # ### CRM, CRMIP
    # Ограничения на оптимизируемые параметры
    # границы для параметров для CRM/CRMIP
    bnds = config.bnds_CRM
    # ---------------------------------------------------------------------------

    minimizer = SciPySlsqpWrapper(options, verbose=config.display_graphs)

    # Выбор модели

    # !!!! Расчет модели. Может занять много времени.
    # Можно не запускать, если есть сохраненная модель в файле crm.pickle
    save_path = config.save_path_train
    models_save_path = config.models_save_path_train

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(models_save_path):
        os.makedirs(models_save_path)

    if not os.path.exists(save_path_output_pipeline_train_valid):
        os.makedirs(save_path_output_pipeline_train_valid)

    tmp_dict = {
        'prod_wells': {},
        'inj_wells': {}
    }

    for (well_name, inj) in injection.iteritems():
        tmp_dict['inj_wells'][well_name] = {'injection': inj[-1]}

    for (well_name, press) in pressure.iteritems():
        tmp_dict['prod_wells'][well_name] = {'pressure': press[-1]}

    for (well_name, prod) in production.iteritems():
        tmp_dict['prod_wells'][well_name]['production'] = prod[-1]

    # save_globals('dump_{}'.format(config.preset_choice))
    # restore_globals('dump_{}'.format(config.preset_choice))
    print('Time mark CRM', datetime.now()-prev_time)

    # построение модели CRM
    model_CRM = CRM(coordinates, bnds, production.columns, injection.columns,
                scheme=SCHEME, tau_0=TAU_0, J_0=J_0, influence_r=INFLUENCE_R, double_r=double_R)
    # обучение модели
    model_CRM.fit(production.values, is_halt, injection.values, rec_pressure.values, minimizer)
    real_f = model_CRM.f.copy() # коэффициент взаимовлияния
    inj_wells_f = np.sum(model_CRM.f, axis=1)

    tmp_ndarray = np.zeros_like(model_CRM.f)
    for i, well in enumerate(model_CRM.inj_n):
        tmp_ndarray[i, :] = model_CRM.f[i, :] if inj_wells_f[i] <= 1 else model_CRM.f[i, :]/inj_wells_f[i]

    model_CRM.f = tmp_ndarray
    model_CRM.to_excel(os.path.join(save_path, 'CRM.xlsx'))

    recounted_inj(inj_wells_f, model_CRM.inj_n).to_excel(os.path.join(save_path, 'recounted_injection_CRM.xlsx'))

    model_CRM.f = real_f.copy()
    print('Time mark', datetime.now()-prev_time)
    concated_df, f_dict = plot.crm_map(pd.DataFrame(model_CRM.f, index=model_CRM.inj_n, columns=model_CRM.prod_n),
                                       coordinates[['Координата X', 'Координата Y']].reset_index(), save_path, 'CRM',
                                       border=config.border)
    plot.crm_plot(concated_df, f_dict, os.path.join(save_path, 'CRM')) # строит карты взаимовлияния CRM

    with open(os.path.join(models_save_path, 'CRM.pickle'), 'wb') as f:
        pickle.dump((tmp_dict, model_CRM), f)

    # Визуализация для CRM

    pred = model_CRM.predict(model_CRM.prod_0, is_halt, injection.values, rec_pressure.values)
    # # print(len(crm.prod_0),crm.prod_0)

    pred_CRM_df = pd.DataFrame(data=pred, index=injection.index, columns=production.columns)
    # pred_train = pd.DataFrame(data=pred_train, index=model.inj_n, columns=production.columns)

    # pred_CRM_df.drop_duаplicates(inplace=True)
    writer = pd.ExcelWriter(os.path.join(save_path, 'CRM_prediction.xlsx'))
    pred_CRM_df.to_excel(writer, 'Sheet1')
    writer.save()

    # save_globals('dump_{}'.format(config.preset_choice))
    print('Time mark CRMIP', datetime.now()-prev_time)
    # построение модели CRMIP
    model_CRMIP = CRMIP(coordinates, bnds, production.columns, injection.columns,
                  scheme='SVIR', tau_0=TAU_0, J_0=J_0, influence_r=INFLUENCE_R, double_r=double_R)

    model_CRMIP.fit(production.values, is_halt, injection.values, rec_pressure.values, minimizer)
    real_f = model_CRMIP.f.copy()
    inj_wells_f = np.sum(model_CRMIP.f, axis=1)

    tmp_ndarray = np.zeros_like(model_CRMIP.f)
    for i, well in enumerate(model_CRMIP.inj_n):
        tmp_ndarray[i, :] = model_CRMIP.f[i, :] if inj_wells_f[i] <= 1 else model_CRMIP.f[i, :] / inj_wells_f[i]

    model_CRMIP.f = tmp_ndarray
    model_CRMIP.to_excel(os.path.join(save_path, 'CRMIP.xlsx'))

    recounted_inj(inj_wells_f, model_CRM.inj_n).to_excel(os.path.join(save_path, 'recounted_injection_CRMIP.xlsx'))

    model_CRMIP.f = real_f.copy()
    print('Time mark', datetime.now()-prev_time)

    concated_df, f_dict = plot.crm_map(pd.DataFrame(model_CRMIP.f, index=model_CRMIP.inj_n, columns=model_CRMIP.prod_n),
                                       coordinates[['Координата X', 'Координата Y']].reset_index(), save_path, 'CRMIP',
                                       border=config.border)
    plot.crm_plot(concated_df, f_dict,  os.path.join(save_path, 'CRMIP')) # строит карты взаимовлияния CRMIP

    with open(os.path.join(models_save_path, 'CRMIP.pickle'), 'wb') as f:
        pickle.dump((tmp_dict, model_CRMIP), f)

    # Визуализация для CRMIP
    pred = model_CRMIP.predict(model_CRMIP.prod_0, is_halt, injection.values, rec_pressure.values)
    pred_CRMIP_df = pd.DataFrame(data=pred, index=injection.index, columns=production.columns)

    # pred_CRMIP_df.drop_duplicates(inplace=True)
    writer = pd.ExcelWriter(os.path.join(save_path, 'CRMIP_prediction.xlsx'))
    pred_CRMIP_df.to_excel(writer, 'Sheet1')
    writer.save()
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # LSTM
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # TODO f только по CRMIP?

    # save_globals('dump_{}'.format(config.preset_choice))
    # restore_globals()
    # restore_globals('dump_{}'.format(config.preset_choice))
    if config.liquid_flag or config.wc_flag or config.gas_factor_flag or config.pressure_reservoir_flag:
        injection_sum = pdInj.prepare_file_with_injection(model_CRMIP.f,
                                                          model_CRMIP.inj_n, model_CRMIP.prod_n,
                                                          injection,
                                                          config.CUT_DATE, config.CUT_DATE2).loc[:, production.columns]
    else:
        injection_sum = pd.DataFrame([])
    # TODO rename
    # columns = [pred_CRM_df, pred_CRMIP_df, pressure, injection_sum]
    columns = [pred_CRMIP_df, pressure, injection_sum]
    name_column_tuples = [(production, 'liquid'), (water_cut, 'wc')]

    df_to_save = [('pressure', pressure), ('injection_sum', injection_sum), ('production', production),
                  ('water_cut', water_cut),
                  ('injection', injection),
                  ('rec_pressure', rec_pressure)]

    for name, df in df_to_save:
        df.to_excel(name+'.xlsx')

    # save_globals('dump_{}_before_tpot'.format(config.preset_choice))
    # restore_globals('dump_{}_before_LSTM'.format(config.preset_choice))
    #
    # TPOT
    # tpot = TPOTRegressor(verbosity=3,
    #                      generations=config.generations,
    #                      population_size=config.population_size,
    #                      n_jobs=-1,
    #                      early_stop=config.early_stop,
    #                      memory=None)
    #
    # all_models_prediction_train_dict = {}
    # all_models_prediction_test_dict = {}
    #
    # all_models_x_true_train_dict = {}
    # all_models_x_true_test_dict = {}
    # # TODO сейчас в model.past могут быть сохранены "лишние"(отсутсвующие в info) скважины
    # tmp_past_set = None
    # for predict_column, name in name_column_tuples:
    #     if vars(globals()['config'])[name+'_flag']:
    #         prepared_data = utils.make_wells_timelines(predict_column, *columns)
    #         model = TimeSeriesModel.TimeSeriesModel(name_=name, test_period_len_=config.test_period_len, model=tpot, mode='tpot')
    #
    #         train_x, train_y, train_dict, test_dict = model.split_data(prepared_data, config.CUT_DATE2)
    #
    #         model.fit(train_x, np.ravel(train_y))
    #
    #         model.save_model(models_save_path_predict=config.models_save_path_train)
    #
    #         # ++++ Train validation ++++ #
    #         all_metrics, predictions_dict, x_true_dict, _ = model.evaluate_model(test_dict, mode='train')
    #
    #         all_models_prediction_train_dict[name] = predictions_dict
    #         all_models_x_true_train_dict[name] = x_true_dict
    #
    #         utils.save_results(os.path.join(save_path_output_pipeline_train_valid, name), name, test_dict, all_metrics,
    #                            predictions_dict, x_true_dict)
    #         # ++++++++++++++++++++++++++ #
    #
    #         # ++++ Test validation ++++ #
    #         all_metrics, predictions_dict, x_true_dict, features_histories = model.evaluate_model(test_dict)
    #
    #         ###########################
    #         fh_path = os.path.join(
    #             os.path.join(config.path_main_train, ), 'feature_histories_' + config.preset_choice,
    #                                                     'feature_histories_' + name)
    #
    #         if not os.path.exists(fh_path):
    #             os.makedirs(fh_path)
    #
    #         for well_name, feature_history in features_histories.items():
    #             pd.DataFrame(feature_history).to_excel(os.path.join(fh_path, well_name + '_feature_history.xlsx'),
    #                                                    index=False)
    #             test_dict[well_name].to_excel(os.path.join(fh_path, well_name + '_feature_true_history.xlsx'),
    #                                           index=False)
    #         ###########################
    #
    #         all_models_prediction_test_dict[name] = predictions_dict
    #         all_models_x_true_test_dict[name] = x_true_dict
    #
    #         utils.save_results(os.path.join(config.save_path_output_pipeline_train, name), name, test_dict, all_metrics,
    #                            predictions_dict, x_true_dict)
    #         # ++++++++++++++++++++++++++ #
    #
    #         if tmp_past_set is None:
    #             tmp_past_set = set(model.past_for_predict.keys())
    #         else:
    #             tmp_past_set = tmp_past_set & set(model.past_for_predict.keys())
    #         print('tmp_past_set', tmp_past_set)
    #
    # if vars(globals()['config'])['liquid_flag'] and vars(globals()['config'])['wc_flag']:
    #     # ++++ Train validation ++++ #
    #     save_path_train_oil = os.path.join(save_path_output_pipeline_train_valid, 'oil')
    #     if not os.path.exists(save_path_train_oil):
    #         os.makedirs(save_path_train_oil)
    #
    #     all_metrics = {}
    #     pred_and_target = []
    #     integr_pred = []
    #     integr_fact = []
    #
    #     # TODO использование model здесь неуместно!
    #     # test_period = model.test_dict[list(model.test_dict.keys())[0]].index
    #     test_period = pd.date_range(start=datetime.strftime(
    #         datetime.strptime(config.CUT_DATE2, '%Y-%m-%d') - timedelta(config.test_period_len - 1)
    #         , '%Y-%m-%d'),
    #         end=config.CUT_DATE2)
    #
    #     liq_wc_wells_names = [well_name for well_name in all_models_prediction_train_dict['wc'].keys()
    #                           if well_name in all_models_prediction_train_dict['liquid']]
    #     for well_name in liq_wc_wells_names:
    #         predictions = (1-all_models_prediction_train_dict['wc'][well_name]) * \
    #                       all_models_prediction_train_dict['liquid'][well_name]
    #
    #         x_true = (1-all_models_x_true_train_dict['wc'][well_name]) * \
    #                  all_models_x_true_train_dict['liquid'][well_name]
    #
    #         metrics = utils.calculate_metrics(predictions, x_true)
    #
    #         for key, value in metrics.items():
    #             if key in all_metrics.keys():
    #                 all_metrics[key].append(value)
    #             else:
    #                 all_metrics[key] = [value]
    #
    #         pdtest = pd.DataFrame(predictions, columns=[well_name + ' Prediction'])
    #         pdhist = pd.DataFrame(x_true, columns=[well_name + ' Target'])
    #         integr_pred.append(pdtest.sum(axis=0))
    #         integr_fact.append(pdhist.sum(axis=0))
    #         pred_and_target.append(pd.concat([pdtest, pdhist], axis=1))
    #         # Get plots predicted vs real for every well
    #         plt.figure(figsize=(16, 5))
    #         plt.plot(test_period, x_true, 'ro-', label='Факт')
    #         plt.plot(test_period, predictions, 'bo-', label='Прогноз')
    #         plt.legend(loc='upper left')
    #         # plt.xlabel('MAE = %.3f' % (float(mae)))
    #         # plt.ylabel('Дебит жидкости')
    #         plt.title('Прогноз скважины номер ' + well_name)
    #         plt.grid(True)
    #         plt.savefig(os.path.join(save_path_train_oil, '{}.png'.format(well_name)))
    #         plt.close()
    #
    #     # write results with real and predicted values into an excel sheet.
    #     pred_and_target = pd.concat(pred_and_target, axis=1)
    #     # TODO убрать костыль
    #     pred_and_target.index = test_period
    #
    #     pd.DataFrame(np.array(integr_pred).flatten(), index=liq_wc_wells_names).to_excel(
    #         os.path.join(save_path_train_oil, '{}_INTEGR_PRED_TEST.xlsx'.format('oil')), header=False)
    #
    #     pd.DataFrame(np.array(integr_fact).flatten(), index=liq_wc_wells_names).to_excel(
    #         os.path.join(save_path_train_oil, '{}_INTEGR_FACT_TEST.xlsx'.format('oil')), header=False)
    #
    #     pred_and_target.to_excel(
    #         os.path.join(save_path_train_oil, '{}_TEST.xlsx'.format('oil')))
    #
    #     for key, value in all_metrics.items():
    #         pd.DataFrame(value, index=liq_wc_wells_names).to_excel(
    #             os.path.join(save_path_train_oil, '{}_TEST_'.format('oil') + key.upper() + '.xlsx'), header=False)
    #
    #     # ++++++++++++++++++++++++++ #
    #
    #     # ++++ Test validation ++++ #
    #     save_path_train_oil = os.path.join(config.save_path_output_pipeline_train, 'oil')
    #     if not os.path.exists(save_path_train_oil):
    #         os.makedirs(save_path_train_oil)
    #
    #     all_metrics = {}
    #     pred_and_target = []
    #     integr_pred = []
    #     integr_fact = []
    #
    #     # TODO использование model здесь неуместно!
    #     # test_period = model.test_dict[list(model.test_dict.keys())[0]].index
    #     test_period = pd.date_range(start=datetime.strftime(
    #         datetime.strptime(config.CUT_DATE2, '%Y-%m-%d') - timedelta(config.test_period_len - 1)
    #         , '%Y-%m-%d'),
    #         end=config.CUT_DATE2)
    #
    #     liq_wc_wells_names = [well_name for well_name in all_models_prediction_test_dict['wc'].keys()
    #                           if well_name in all_models_prediction_test_dict['liquid']]
    #     for well_name in liq_wc_wells_names:
    #         predictions = (1 - all_models_prediction_test_dict['wc'][well_name]) * \
    #                       all_models_prediction_test_dict['liquid'][well_name]
    #
    #         x_true = (1 - all_models_x_true_test_dict['wc'][well_name]) \
    #                  * all_models_x_true_test_dict['liquid'][well_name]
    #
    #         metrics = utils.calculate_metrics(predictions, x_true)
    #
    #         for key, value in metrics.items():
    #             if key in all_metrics.keys():
    #                 all_metrics[key].append(value)
    #             else:
    #                 all_metrics[key] = [value]
    #
    #         pdtest = pd.DataFrame(predictions, columns=[well_name + ' Prediction'])
    #         pdhist = pd.DataFrame(x_true, columns=[well_name + ' Target'])
    #         integr_pred.append(pdtest.sum(axis=0))
    #         integr_fact.append(pdhist.sum(axis=0))
    #         pred_and_target.append(pd.concat([pdtest, pdhist], axis=1))
    #         # Get plots predicted vs real for every well
    #         plt.figure(figsize=(16, 5))
    #         plt.plot(test_period, x_true, 'ro-', label='Факт')
    #         plt.plot(test_period, predictions, 'bo-', label='Прогноз')
    #         plt.legend(loc='upper left')
    #         # plt.xlabel('MAE = %.3f' % (float(mae)))
    #         # plt.ylabel('Дебит жидкости')
    #         plt.title('Прогноз скважины номер ' + well_name)
    #         plt.grid(True)
    #         plt.savefig(os.path.join(save_path_train_oil, '{}.png'.format(well_name)))
    #         plt.close()
    #
    #     # write results with real and predicted values into an excel sheet.
    #     pred_and_target = pd.concat(pred_and_target, axis=1)
    #     # TODO убрать костыль
    #     pred_and_target.index = test_period
    #
    #     pd.DataFrame(np.array(integr_pred).flatten(), index=liq_wc_wells_names).to_excel(
    #         os.path.join(save_path_train_oil, '{}_INTEGR_PRED_TEST.xlsx'.format('oil')), header=False)
    #
    #     pd.DataFrame(np.array(integr_fact).flatten(), index=liq_wc_wells_names).to_excel(
    #         os.path.join(save_path_train_oil, '{}_INTEGR_FACT_TEST.xlsx'.format('oil')), header=False)
    #
    #     pred_and_target.to_excel(
    #         os.path.join(save_path_train_oil, '{}_TEST.xlsx'.format('oil')))
    #
    #     for key, value in all_metrics.items():
    #         pd.DataFrame(value, index=liq_wc_wells_names).to_excel(
    #             os.path.join(save_path_train_oil, '{}_TEST_'.format('oil') + key.upper() + '.xlsx'), header=False)

        # ++++++++++++++++++++++++++ #
    # Service
    tmp_past_set = None
    info = get_info(injection, rec_pressure, tmp_past_set)

    if info is not None:
        with open(os.path.join(models_save_path, 'info.json'), 'w') as f:
            json.dump(info, f, indent=4, sort_keys=False)

        with open(os.path.join(models_save_path, 'config.json'), 'w') as f:
            json.dump({'p_res': config.p_res,
                       'p_0': config.p_0,
                       'recov_time': config.recov_time,
                       'density': 0.8
                       }, f)

    print('Time mark', datetime.now()-prev_time)
    print('finished')


if __name__ == '__main__':
    main()
