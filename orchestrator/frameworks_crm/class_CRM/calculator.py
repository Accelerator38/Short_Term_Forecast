import logging.config
from datetime import datetime, timedelta
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from frameworks_crm.class_CRM.config import ConfigCRM
from frameworks_crm.class_CRM.data_processor_crm import DataProcessorCRM
from frameworks_crm.crm.models.crm import CRM
from frameworks_crm.crm.models.crmip import CRMIP
from frameworks_crm.crm.optimization.minimizers import SciPySlsqpWrapper
from frameworks_crm.crm.preprocessing.utils import cut_inactive_wells
from frameworks_crm.crm.preprocessing.utils import recover_pres
from frameworks_crm.crm.preprocessing.utils import select_wells_nearest_to_prod


class Calculator:
    def __init__(self, config: ConfigCRM):
        self._logger = logging.getLogger('crm.__main__')
        self._injection = None
        self._CRM = None
        self._CRMIP = None
        self._config_from_train = None
        self._info = None

        self._config = config
        self._data_processor = DataProcessorCRM(self._config)
        self._get_input_data(self._data_processor)
        self._train()
        if self._config.calc_CRM:
            self._predict(date=self._config.date_end_forecast, mode='crm')
        if self._config.calc_CRMIP:
            self._predict(date=self._config.date_end_forecast, mode='crmip')

    def _get_input_data(self, data_processor):
        self._prod_wells = data_processor.prod_wells
        self._inj_wells = data_processor.inj_wells
        self._coordinates = data_processor.coordinates
        self._governm_prod = data_processor.governm_prod
        self._governm_inj = data_processor.governm_inj

    def _train(self):
        log_config_path = Path(__file__).parent / 'logging.conf'
        path_to_save_config = str(Path(__file__).parent / 'crm.log')
        path_to_save_config = path_to_save_config.replace('\\', '\\\\')
        logging.config.fileConfig(log_config_path, defaults={'logfilepath': path_to_save_config})
        self._logger.info('INFLUENCE_R ' + str(self._config.INFLUENCE_R))
        self._logger.info('double_R ' + str(self._config.double_R))
        self._logger.info('inactive_time ' + str(self._config.inactive_time))
        self._logger.info('CUT_DATE ' + str(self._config.CUT_DATE))
        self._logger.info('CUT_DATE2 ' + str(self._config.CUT_DATE2))
        self._logger.info('p_res ' + str(self._config.p_res))
        self._logger.info('p_0 ' + str(self._config.p_0))
        self._logger.info('recov_time ' + str(self._config.recov_time))
        self._logger.info('TAU_0 ' + str(self._config.TAU_0))
        self._logger.info('J_0 ' + str(self._config.J_0))
        self._logger.info('options_SLSQP_CRM ' + str(self._config.options_SLSQP_CRM))
        self._logger.info('liquid_flag ' + str(self._config.liquid_flag))
        self._logger.info('wc_flag ' + str(self._config.wc_flag))
        self._logger.info('gas_factor_flag ' + str(self._config.gas_factor_flag))
        self._logger.info('pressure_reservoir_flag ' + str(self._config.pressure_reservoir_flag))
        self._logger.info('prod_bnd ' + str(self._config.prod_bnd))
        self._logger.info('tau_bnd ' + str(self._config.tau_bnd))
        self._logger.info('J_bnd ' + str(self._config.J_bnd))
        self._logger.info('f_bnd ' + str(self._config.f_bnd))

        # self._config.save_path_train = os.path.join(self._config.save_path_output_pipeline_train, 'CRM_output')
        self._config.bnds_CRM = {'prod_bnd': self._config.prod_bnd, 'tau_bnd': self._config.tau_bnd,
                                 'J_bnd': self._config.J_bnd, 'f_bnd': self._config.f_bnd}

        # save_path_output_pipeline_train_valid = os.path.join(self._config.save_path_output_pipeline_train,
        # 'train_validation') self._config.test_period_len = ((1 - self._config.split_ratio) * (datetime.strptime(
        # self._config.CUT_DATE2, '%Y-%m-%d') - datetime.strptime(self._config.CUT_DATE, '%Y-%m-%d'))).days - 1
        # Prepare data to CRM +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        prev_time = datetime.now()

        print('Time mark', datetime.now() - prev_time)
        # xls = pd.ExcelFile(
        #     os.path.join(self._config.path_train,
        #                  f'Data_model_reservoir_{self._config.oilfield}.xlsx'))  # данные с месторождения
        # coordinates = pd.read_excel(xls, sheet_name=2, dtype={'Скважина': str})
        coordinates = self._coordinates
        if not self._config.double_R:
            coordinates = coordinates[['Координата X', 'Координата Y']]
        print('Time mark', datetime.now() - prev_time)

        # nef_info = pd.read_excel(xls, sheet_name=0, dtype={'Скважина': str})
        nef_info = self._prod_wells
        print('Time mark', datetime.now() - prev_time)
        production = pd.pivot_table(nef_info, values='Дебит жидкости',
                                    index='Дата', columns='Скважина', dropna=False) \
                         .apply(lambda x: x.interpolate(limit_direction='forward', limit_area='inside'), axis=0) \
            [self._config.CUT_DATE:self._config.CUT_DATE2].dropna(axis=1, how='all').fillna(0)
        # заполняет пропуски дат в данных дебитах жидкости
        production.columns.name = None

        print('Time mark', datetime.now() - prev_time)
        water_cut = pd. \
                        pivot_table(nef_info, values='Обводненность', index='Дата', columns='Скважина', dropna=False). \
                        apply(lambda x: x.interpolate(limit_direction='forward', limit_area='inside'), axis=0)[
                    self._config.CUT_DATE:self._config.CUT_DATE2]. \
            dropna(axis=1, how='all')  # заполняет пропуски дат в данных обводненности
        water_cut.columns.name = None

        print('Time mark', datetime.now() - prev_time)
        pressure = pd. \
                       pivot_table(nef_info, values='Забойное давление', index='Дата', columns='Скважина',
                                   dropna=False). \
                       apply(lambda x: x.interpolate(limit_direction='forward', limit_area='inside'), axis=0)[
                   self._config.CUT_DATE:self._config.CUT_DATE2]. \
            dropna(axis=1, how='all').fillna(0)  # заполняет пропуски дат в данных забойного давления
        pressure.columns.name = None

        print('Time mark', datetime.now() - prev_time)
        print('_______')

        # nag_info = pd.read_excel(xls, sheet_name=1, dtype={'Скважина': str})
        nag_info = self._inj_wells
        print('Time mark', datetime.now() - prev_time)
        self._injection = pd. \
                              pivot_table(nag_info, values='Приемистость', index='Дата', columns='Скважина',
                                          dropna=False). \
                              apply(lambda x: x.interpolate(limit_direction='forward', limit_area='inside'), axis=0)[
                          self._config.CUT_DATE:self._config.CUT_DATE2]. \
            dropna(axis=1, how='all').fillna(0)  # заполняет пропуски дат в данных приемистости
        self._injection.columns.name = None

        print('Time mark', datetime.now() - prev_time)

        # индекс в datetime переводит
        production.index = pd.to_datetime(production.index)
        pressure.index = pd.to_datetime(pressure.index)
        water_cut.index = pd.to_datetime(water_cut.index)
        self._injection.index = pd.to_datetime(self._injection.index)

        # меняет граничные даты, если в self._config указаны даты не входящие в данные
        list([self._change_cut_dates_nes(self._config, el.index[0], el.index[-1]) for el in
              (production, water_cut, pressure, self._injection)])
        production = production.loc[self._config.CUT_DATE:self._config.CUT_DATE2]
        water_cut = water_cut.loc[self._config.CUT_DATE:self._config.CUT_DATE2]
        pressure = pressure.loc[self._config.CUT_DATE:self._config.CUT_DATE2]
        self._injection = self._injection.loc[self._config.CUT_DATE:self._config.CUT_DATE2]

        # делает переиндексацию на новые даты. Кажется, что можно без этой процедуры
        ind = pd.date_range(start=production.index[0], end=production.index[-1], freq='D')
        production = production.reindex(ind). \
            apply(lambda x: x.interpolate(method='linear', limit_direction='forward', limit_area='inside'), axis=0)

        pressure = pressure.reindex(ind). \
            apply(lambda x: x.interpolate(method='linear', limit_direction='forward', limit_area='inside'), axis=0)

        water_cut = water_cut.reindex(ind). \
            apply(lambda x: x.interpolate(method='linear', limit_direction='forward', limit_area='inside'), axis=0)

        self._injection = self._injection.reindex(ind). \
            apply(lambda x: x.interpolate(method='linear', limit_direction='forward', limit_area='inside'), axis=0)

        name_inj_wells = list(self._injection.columns)

        prod_df_lst = [production, pressure, water_cut]
        names_prod_wells = list(reduce(lambda x, y: x & set(y.columns), prod_df_lst[1:],
                                       set(prod_df_lst[0].columns))  # названия добывающих скважин
                                - set(name_inj_wells))

        for i, df in enumerate(prod_df_lst):
            prod_df_lst[i] = df[names_prod_wells]

            # в names_prod_wells нет и нагнетающих скважин,
            # и скважин, которых нет хотя бы в одной их характеристик добывающих скважин.
            # однако, непонятно, как информация о переводах вносится в приемистость.
            # поэтому необходимо занулить приемистость "из прошлого".
            # НЕ НУЖНО, ПОТОМУ ЧТО СЕЙЧАС наны в приемистости заполняются нулями и так. код с ошибками.
            # self._injection[(set(df.columns) - set(names_prod_wells)) & set(name_inj_wells)] = 0

        production, pressure, water_cut = prod_df_lst

        print(production.shape)
        print(pressure.shape)
        print(water_cut.shape)
        print(self._injection.shape)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # coordinates.set_index(coordinates.columns[0], inplace=True)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # read_options = dict(index_col=0, parse_dates=True)
        # path = self._config.path_train

        coordinates = coordinates[~coordinates.index.duplicated(keep='first')]

        # Матрицы, когда для дебита или приемистости есть соответствующее значение забойного давления (давления в линии)
        # prod = set(self._injection.columns.values)

        # Скважины, которые простаивают большую часть времени необходимо исключить,
        # так как они вносят лишнюю неопределенность.
        active_prod_wells = cut_inactive_wells(production, thres=self._config.inactive_time)[0]  # активные скважины
        production = production.loc[:, active_prod_wells.columns]
        pressure = pressure.loc[:, active_prod_wells.columns]
        water_cut = water_cut.loc[:, active_prod_wells.columns]

        prod = set(production.columns.values)
        pres = set(pressure.columns.values)
        list1 = list(pres & prod)
        production = production.loc[:, list1]
        pressure = pressure.loc[:, list1]
        # print(len(production.columns), len(pressure.columns))

        production = production.loc[:, ~production.columns.duplicated(keep='first')]
        pressure = pressure.loc[:, ~pressure.columns.duplicated(keep='first')]
        water_cut = water_cut.loc[:, ~water_cut.columns.duplicated(keep='first')]
        self._injection = self._injection.loc[:, ~self._injection.columns.duplicated(keep='first')]
        # print(len(production.columns), len(pressure.columns))

        # Нагнетательные скважины, удаленные на большое расстояние от добывающих вносят незначительный вклад либо вовсе
        # незначимы, поэтому для уменьшения размерности задачи их необходимо отбросить.
        names_prod_wells = list(set(production.columns) & set(coordinates.index))
        production = production[names_prod_wells]
        pressure = pressure[names_prod_wells]
        water_cut = water_cut[names_prod_wells]

        names_inj_wells = list(set(self._injection.columns) & set(coordinates.index))
        self._injection = self._injection[names_inj_wells]
        self._injection = select_wells_nearest_to_prod(self._injection, production, coordinates,
                                                       dist_thres=self._config.INFLUENCE_R,
                                                       double_R=self._config.double_R)

        # Для более точной аппроксимации значений дебитов при остановке и включении скважин, необходимо заполнить
        # нулевые значения для давлений. Для этого нужно знать значение пластового давления и характерное время
        # восстановления давления. d1 =set(production.columns.values).difference(set(pressure.columns.values))
        # print(set(production.columns.values)) # print(pressure.info()) # print(production.info())
        rec_pressure = recover_pres(pressure, inplace=False, p_res=self._config.p_res, p_0=self._config.p_0,
                                    recov_time=self._config.recov_time)  # заполняет значения в забойном давлении

        is_work = production.astype(bool).astype(int)
        is_halt = np.array(np.ones_like(is_work) - is_work, dtype=np.int64)

        # # Оптимизация модели
        # Для моделей CRM/CRMIP
        # схема модели svir/lwir
        SCHEME = self._config.SCHEME
        # начальное значение параметров модели
        TAU_0 = self._config.TAU_0
        J_0 = self._config.J_0
        # радиус влияния (вне его скважины считаются невзаимодействующими)
        INFLUENCE_R = self._config.INFLUENCE_R
        double_R = self._config.double_R

        # Параметры оптимизации SLSQP
        options = self._config.options_SLSQP_CRM

        # ### CRM, CRMIP
        # Ограничения на оптимизируемые параметры
        # границы для параметров для CRM/CRMIP
        bnds = self._config.bnds_CRM
        # ---------------------------------------------------------------------------

        minimizer = SciPySlsqpWrapper(options)

        # Выбор модели

        # !!!! Расчет модели. Может занять много времени.
        # Можно не запускать, если есть сохраненная модель в файле crm.pickle
        # save_path = self._config.save_path_train
        # models_save_path = self._config.models_save_path_train
        #
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        #
        # if not os.path.exists(models_save_path):
        #     os.makedirs(models_save_path)
        #
        # if not os.path.exists(save_path_output_pipeline_train_valid):
        #     os.makedirs(save_path_output_pipeline_train_valid)

        tmp_dict = {
            'prod_wells': {},
            'inj_wells': {}
        }

        for (well_name, inj) in self._injection.items():
            tmp_dict['inj_wells'][well_name] = {'injection': inj[-1]}

        for (well_name, press) in pressure.items():
            tmp_dict['prod_wells'][well_name] = {'pressure': press[-1]}

        for (well_name, prod) in production.items():
            tmp_dict['prod_wells'][well_name]['production'] = prod[-1]

        # save_globals('dump_{}'.format(self._config.preset_choice))
        # restore_globals('dump_{}'.format(self._config.preset_choice))
        print('Time mark CRM', datetime.now() - prev_time)

        # построение модели CRM
        if self._config.calc_CRM:
            model_CRM = CRM(coordinates, bnds, production.columns, self._injection.columns,
                            scheme=SCHEME, tau_0=TAU_0, J_0=J_0, influence_r=INFLUENCE_R, double_r=double_R)
            # обучение модели
            model_CRM.fit(production.values, is_halt, self._injection.values, rec_pressure.values, minimizer)
            real_f = model_CRM.f.copy()  # коэффициент взаимовлияния
            inj_wells_f = np.sum(model_CRM.f, axis=1)

            tmp_ndarray = np.zeros_like(model_CRM.f)
            for i, well in enumerate(model_CRM.inj_n):
                tmp_ndarray[i, :] = model_CRM.f[i, :] if inj_wells_f[i] <= 1 else model_CRM.f[i, :] / inj_wells_f[i]

            model_CRM.f = tmp_ndarray
            self._f = pd.DataFrame(model_CRM.f, index=model_CRM.inj_n, columns=model_CRM.prod_n)
            self._tau = pd.DataFrame(model_CRM.tau, index=model_CRM.prod_n, columns=['tau'])
            self._J = pd.DataFrame(model_CRM.J, index=model_CRM.prod_n, columns=['J'])
            self._tau.index.name = 'Well'
            self._J.index.name = 'Well'
            # model_CRM.to_excel(str(Path(__file__).parent / 'CRM_params.xlsx'))

            # self._recounted_inj(inj_wells_f, model_CRM.inj_n).to_excel(
            #     os.path.join(save_path, 'recounted_injection_CRM.xlsx'))

            model_CRM.f = real_f.copy()
            print('Time mark', datetime.now() - prev_time)
            # concated_df, f_dict = plot.crm_map(pd.DataFrame(model_CRM.f, index=model_CRM.inj_n, columns=model_CRM.prod_n),
            #                                    coordinates.reset_index(), save_path, 'CRM',
            #                                    border=self._config.border)
            # plot.crm_plot(concated_df, f_dict, os.path.join(save_path, 'CRM'))  # строит карты взаимовлияния CRM

            self._CRM = (tmp_dict, model_CRM)
        # with open(os.path.join(models_save_path, 'CRM.pickle'), 'wb') as f:
        #     pickle.dump((tmp_dict, model_CRM), f)

        # Визуализация для CRM

        # pred = model_CRM.predict(model_CRM.prod_0, is_halt, self._injection.values, rec_pressure.values)
        # # print(len(crm.prod_0),crm.prod_0)

        # pred_CRM_df = pd.DataFrame(data=pred, index=self._injection.index, columns=production.columns)
        # pred_train = pd.DataFrame(data=pred_train, index=model.inj_n, columns=production.columns)

        # pred_CRM_df.drop_duаplicates(inplace=True)
        # writer = pd.ExcelWriter(os.path.join(save_path, 'CRM_prediction.xlsx'))
        # pred_CRM_df.to_excel(writer, 'Sheet1')
        # writer.save()

        # save_globals('dump_{}'.format(self._config.preset_choice))
        print('Time mark CRMIP', datetime.now() - prev_time)

        # построение модели CRMIP
        if self._config.calc_CRMIP:
            model_CRMIP = CRMIP(coordinates, bnds, production.columns, self._injection.columns,
                                scheme='SVIR', tau_0=TAU_0, J_0=J_0, influence_r=INFLUENCE_R, double_r=double_R)

            model_CRMIP.fit(production.values, is_halt, self._injection.values, rec_pressure.values, minimizer)
            real_f = model_CRMIP.f.copy()
            inj_wells_f = np.sum(model_CRMIP.f, axis=1)

            tmp_ndarray = np.zeros_like(model_CRMIP.f)
            for i, well in enumerate(model_CRMIP.inj_n):
                tmp_ndarray[i, :] = model_CRMIP.f[i, :] if inj_wells_f[i] <= 1 else model_CRMIP.f[i, :] / inj_wells_f[i]

            model_CRMIP.f = tmp_ndarray
            # model_CRMIP.to_excel(os.path.join(save_path, 'CRMIP.xlsx'))

            # self._recounted_inj(inj_wells_f, model_CRM.inj_n).to_excel(os.path.join(save_path, 'recounted_injection_CRMIP.xlsx'))

            model_CRMIP.f = real_f.copy()
            print('Time mark', datetime.now() - prev_time)

            # concated_df, f_dict = plot.crm_map(
            #     pd.DataFrame(model_CRMIP.f, index=model_CRMIP.inj_n, columns=model_CRMIP.prod_n),
            #     coordinates.reset_index(), save_path, 'CRMIP',
            #     border=self._config.border)
            # plot.crm_plot(concated_df, f_dict, os.path.join(save_path, 'CRMIP'))  # строит карты взаимовлияния CRMIP

            self._CRMIP = (tmp_dict, model_CRMIP)
            # with open(os.path.join(models_save_path, 'CRMIP.pickle'), 'wb') as f:
            #     pickle.dump((tmp_dict, model_CRMIP), f)

            # Визуализация для CRMIP
            # pred = model_CRMIP.predict(model_CRMIP.prod_0, is_halt, self._injection.values, rec_pressure.values)
            # pred_CRMIP_df = pd.DataFrame(data=pred, index=self._injection.index, columns=production.columns)

            # pred_CRMIP_df.drop_duplicates(inplace=True)
            # writer = pd.ExcelWriter(os.path.join(save_path, 'CRMIP_prediction.xlsx'))
            # pred_CRMIP_df.to_excel(writer, 'Sheet1')
            # writer.save()
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # LSTM
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # TODO f только по CRMIP?

        # save_globals('dump_{}'.format(self._config.preset_choice))
        # restore_globals()
        # restore_globals('dump_{}'.format(self._config.preset_choice))
        # if (self._config.liquid_flag
        #         or self._config.wc_flag
        #         or self._config.gas_factor_flag
        #         or self._config.pressure_reservoir_flag):
        #     injection_sum = pdInj.prepare_file_with_injection(model_CRMIP.f,
        #                                                       model_CRMIP.inj_n, model_CRMIP.prod_n,
        #                                                       self._injection,
        #                                                       self._config.CUT_DATE,
        #                                                       self._config.CUT_DATE2).loc[:, production.columns]
        # else:
        #     injection_sum = pd.DataFrame([])
        # # TODO rename
        # # columns = [pred_CRM_df, pred_CRMIP_df, pressure, injection_sum]
        # columns = [pred_CRMIP_df, pressure, injection_sum]
        # name_column_tuples = [(production, 'liquid'), (water_cut, 'wc')]
        #
        # df_to_save = [('pressure', pressure), ('injection_sum', injection_sum), ('production', production),
        #               ('water_cut', water_cut),
        #               ('injection', self._injection),
        #               ('rec_pressure', rec_pressure)]

        # for name, df in df_to_save:
        #     df.to_excel(name + '.xlsx')

        # save_globals('dump_{}_before_tpot'.format(self._config.preset_choice))
        # restore_globals('dump_{}_before_LSTM'.format(self._config.preset_choice))
        #
        # TPOT
        # tpot = TPOTRegressor(verbosity=3,
        #                      generations=self._config.generations,
        #                      population_size=self._config.population_size,
        #                      n_jobs=-1,
        #                      early_stop=self._config.early_stop,
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
        #     if vars(globals()['self._config'])[name+'_flag']:
        #         prepared_data = utils.make_wells_timelines(predict_column, *columns)
        #         model = TimeSeriesModel.TimeSeriesModel(name_=name, test_period_len_=self._config.test_period_len, model=tpot, mode='tpot')
        #
        #         train_x, train_y, train_dict, test_dict = model.split_data(prepared_data, self._config.CUT_DATE2)
        #
        #         model.fit(train_x, np.ravel(train_y))
        #
        #         model.save_model(models_save_path_predict=self._config.models_save_path_train)
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
        #             os.path.join(self._config.path_main_train, ), 'feature_histories_' + self._config.preset_choice,
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
        #         utils.save_results(os.path.join(self._config.save_path_output_pipeline_train, name), name, test_dict, all_metrics,
        #                            predictions_dict, x_true_dict)
        #         # ++++++++++++++++++++++++++ #
        #
        #         if tmp_past_set is None:
        #             tmp_past_set = set(model.past_for_predict.keys())
        #         else:
        #             tmp_past_set = tmp_past_set & set(model.past_for_predict.keys())
        #         print('tmp_past_set', tmp_past_set)
        #
        # if vars(globals()['self._config'])['liquid_flag'] and vars(globals()['self._config'])['wc_flag']:
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
        #         datetime.strptime(self._config.CUT_DATE2, '%Y-%m-%d') - timedelta(self._config.test_period_len - 1)
        #         , '%Y-%m-%d'),
        #         end=self._config.CUT_DATE2)
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
        #     save_path_train_oil = os.path.join(self._config.save_path_output_pipeline_train, 'oil')
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
        #         datetime.strptime(self._config.CUT_DATE2, '%Y-%m-%d') - timedelta(self._config.test_period_len - 1)
        #         , '%Y-%m-%d'),
        #         end=self._config.CUT_DATE2)
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
        self._info = self._get_info(self._injection, rec_pressure)

        # if info is not None:
        #     with open(os.path.join(models_save_path, 'info.json'), 'w') as f:
        #         json.dump(info, f, indent=4, sort_keys=False)

        self._config_from_train = {'p_res': self._config.p_res,
                                   'p_0': self._config.p_0,
                                   'recov_time': self._config.recov_time,
                                   'density': 0.8
                                   }
        # with open(os.path.join(models_save_path, 'config.json'), 'w') as f:
        #     json.dump({'p_res': self._config.p_res,
        #                'p_0': self._config.p_0,
        #                'recov_time': self._config.recov_time,
        #                'density': 0.8
        #                }, f)

        print('Time mark', datetime.now() - prev_time)
        print('finished')

    def _change_cut_dates_nes(self, config, ind_left, ind_right):
        """
        Меняет даты если, не входят в CONFIG
        """
        print(ind_left, ind_right)
        if config.CUT_DATE < ind_left.date():
            config.CUT_DATE = datetime.strftime(ind_left, '%Y-%m-%d')
            self._logger.info('CUT_DATE changed')
            print('CUT_DATE changed')

        if config.CUT_DATE2 > ind_right.date():
            config.CUT_DATE2 = datetime.strftime(ind_right, '%Y-%m-%d')
            self._logger.info('CUT_DATE2 changed')
            print('CUT_DATE2 changed')

    def _recounted_inj(self, inj_wells_f, inj_names):
        """
        Пересчитывает приемистость умножая её на коэффициент взаимовлияния
        :param inj_wells_f: коэффициент взаимовлияния суммарный для скважины
        :param inj_names: имена нагнетающих скважин
        :return: DateFrame с пересчитанной приемистостью
        """
        # TODO rename
        inj_recounted_dict = {}
        # TODO переделать
        for i, well_name in enumerate(inj_names):
            inj_recounted_dict[well_name] = inj_wells_f[i] * self._injection[well_name]
        inj_recounted_df = pd.DataFrame(inj_recounted_dict)

        return inj_recounted_df

    def _predict(self, date=None, delta=30, pressure=None, injection=None,
                 j_CRM=None, j_CRMIP=None, mode='crm'):
        info = self._info
        config = self._config_from_train

        PREDICTION_DATE = info['restart_date']
        if date:
            PREDICTION_DATE2 = date
        elif delta:
            PREDICTION_DATE2 = datetime.strftime(
                datetime.strptime(PREDICTION_DATE, '%Y-%m-%d') + timedelta(delta), '%Y-%m-%d')
        else:
            raise ValueError('You must specify delta or date')

        date_range = pd.date_range(start=PREDICTION_DATE, end=PREDICTION_DATE2)

        empty_df = pd.DataFrame([], index=date_range)
        if isinstance(pressure, list):
            tmp_pressure = pd.DataFrame([pressure] * len(date_range), index=date_range)
        elif pressure is None:
            tmp_pressure = empty_df
        else:
            tmp_pressure = pressure

        if isinstance(injection, list):
            tmp_injection = pd.DataFrame([injection] * len(date_range), index=date_range)
        elif injection is None:
            tmp_injection = empty_df
        else:
            tmp_injection = injection

        tmp_pressure = tmp_pressure[PREDICTION_DATE:PREDICTION_DATE2]
        tmp_injection = tmp_injection[PREDICTION_DATE:PREDICTION_DATE2]

        # if mode == 'ml':
        #     predictions, pressure, injection = \
        #         self._predict_specific_model(info, (config['p_res'], config['p_0'], config['recov_time']),
        #                                      tmp_pressure, tmp_injection,
        #                                      j_CRM, j_CRMIP, mode=mode)
        #
        #     print('finished')
        #     return predictions, pressure, injection
        if mode in ('crm', 'crmip'):
            self._predict_specific_model(info, (config['p_res'], config['p_0'], config['recov_time']),
                                         tmp_pressure, tmp_injection,
                                         j_CRM, j_CRMIP, mode=mode)

            print('finished')
        else:
            raise ValueError('Wrong mode! Available variants are crm, crmip')

    def _predict_specific_model(self, info, recover_pressure_params, pressure=None, injection=None,
                                j_CRM=None, j_CRMIP=None, mode='ml'):

        PREDICTION_DATE = pressure.index[0]
        PREDICTION_DATE2 = pressure.index[-1]

        print(PREDICTION_DATE, PREDICTION_DATE2)
        date_range = pd.date_range(start=PREDICTION_DATE, end=PREDICTION_DATE2)
        empty_df = pd.DataFrame([], index=date_range)
        if pressure is None:
            pressure = empty_df
        if injection is None:
            injection = empty_df
        crm_wells_dict, CRM_model = self._CRM
        if j_CRM is not None:
            CRM_model.J = j_CRM.loc[CRM_model.prod_n].values.flatten()

        inj_wells_names_CRM = list(crm_wells_dict['inj_wells'].keys())
        prod_wells_names_CRM = list(crm_wells_dict['prod_wells'].keys())

        # add uncontrollable wells
        tmp_dict = {}
        for well_name in set(inj_wells_names_CRM) - set(injection.columns):
            tmp_dict[well_name] = [crm_wells_dict['inj_wells'][well_name]['injection']] * len(injection.index)
        injection = pd.concat([injection, pd.DataFrame(data=tmp_dict, index=date_range)], axis=1)

        tmp_dict = {}
        for well_name in set(prod_wells_names_CRM) - set(pressure.columns):
            tmp_dict[well_name] = [crm_wells_dict['prod_wells'][well_name]['pressure']] * len(pressure.index)
        pressure = pd.concat([pressure, pd.DataFrame(data=tmp_dict, index=date_range)], axis=1)

        tmp_dict = {}
        for well_name in prod_wells_names_CRM:
            tmp_dict[well_name] = [crm_wells_dict['prod_wells'][well_name]['production']]
        production = pd.DataFrame(data=tmp_dict, index=date_range)
        # set wells in the right order
        injection = injection.loc[:, inj_wells_names_CRM]
        pressure = pressure.loc[:, prod_wells_names_CRM]
        # Для более точной аппроксимации значений дебитов при остановке и включении скважин, необходимо заполнить
        # нулевые значения для давлений. Для этого нужно знать значение пластового давления и характерное время
        # восстановления давления.
        rec_pressure = recover_pres(pressure, False, *recover_pressure_params)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        #
        # if save_outputs:
        #     injection.to_excel(os.path.join(save_path, 'injection_result.xlsx'))
        #     pressure.to_excel(os.path.join(save_path, 'pressure_result.xlsx'))
        #     rec_pressure.to_excel(os.path.join(save_path, 'rec_pressure_result.xlsx'))
        # TODO подумать
        # is_work = pressure.astype(bool).astype(int)
        # is_halt = np.array(np.ones_like(is_work) - is_work, dtype=np.int64)
        is_halt = np.zeros(rec_pressure.shape, dtype=np.int64)

        if mode in ('crm', 'ml'):
            # CRM
            prod_0 = CRM_model.pl.split(CRM_model._initial_guess(production.values))[0]
            pred_CRM = CRM_model.predict(prod_0, is_halt, injection.values, rec_pressure.values)
            self._pred_CRM = pd.DataFrame(data=pred_CRM, index=injection.index, columns=pressure.columns)
            # if save_outputs:
            #     pred_CRM.to_excel(os.path.join(save_path, 'CRM_pred.xlsx'))
        elif mode in ('crmip', 'ml'):
            _, CRMIP_model = self._CRMIP
            if j_CRMIP is not None:
                CRMIP_model.J = j_CRMIP.loc[CRMIP_model.inj_n, CRMIP_model.prod_n].values
            # CRMIP
            prod_0 = CRMIP_model.rsc.convert(CRMIP_model.pl.split(CRMIP_model._initial_guess(production.values))[0])
            pred_CRMIP = CRMIP_model.predict(prod_0, is_halt, injection.values, rec_pressure.values)
            self._pred_CRMIP = pd.DataFrame(data=pred_CRMIP, index=injection.index, columns=pressure.columns)
            # if save_outputs:
            #     pred_CRMIP.to_excel(os.path.join(save_path, 'CRMIP_pred.xlsx'))

        # if mode == 'ml':
        #     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #     # LSTM
        #     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #     # TODO f только по CRMIP?
        #     injection_sum = pdInj.prepare_file_with_injection(CRMIP_model.f, CRMIP_model.inj_n, CRMIP_model.prod_n,
        #                                                       injection,
        #                                                       info['restart_date'], PREDICTION_DATE2)
        #
        #     # throw out bad wells
        #     pred_CRM = pred_CRM.loc[:, prod_wells_names_LSTM]
        #     pred_CRMIP = pred_CRMIP.loc[:, prod_wells_names_LSTM]
        #     injection_sum = injection_sum.loc[:, prod_wells_names_LSTM]
        #
        #     if verbose:
        #         print('Time mark 1', datetime.now() - prev_time)
        #
        #     predictions = {}
        #     tmp_dict = {}
        #     for well_name in pred_CRM.columns:
        #         tmp_dict[well_name] = np.zeros((pred_CRM[well_name].values.shape[0]))
        #
        #     zeros_df = pd.DataFrame(tmp_dict, index=pd.date_range(start=PREDICTION_DATE, end=PREDICTION_DATE2))
        #
        #     # prepared_data = utils.make_wells_timelines(zeros_df, pred_CRM, pred_CRMIP, rec_pressure, injection_sum)
        #     prepared_data = utils.make_wells_timelines(zeros_df, pred_CRMIP, rec_pressure, injection_sum)
        #
        #     models_name = [dir_name for dir_name in os.listdir(models_path)
        #                    if not os.path.isfile(os.path.join(models_path, dir_name))]
        #
        #     for i, name in enumerate(models_name):
        #         if flags.get(name + '_flag', True):
        #
        #             if verbose:
        #                 prev_time_2 = datetime.now()
        #                 print('Time mark', name, '1', datetime.now() - prev_time_2)
        #
        #             # TODO del in future. models incompatibility
        #             with open(os.path.join(os.path.join(models_path, name), 'model.pickle'), 'rb') as f:
        #                 tmp_str = re.sub(b'TimeSeriesModel\nTimeSeriesModel', b'hybrid.Base_TSF_Model\nTimeSeriesModel',
        #                                  f.read())
        #
        #             with open(os.path.join(os.path.join(models_path, name), 'model.pickle'), 'wb') as f:
        #                 f.write(tmp_str)
        #
        #             model = utils.get_pickle(os.path.join(models_path, name), 'model')
        #
        #             if verbose:
        #                 print('Time mark', name, '2', datetime.now() - prev_time_2)
        #
        #             predictions[name + '_orig'] = model.predict(prepared_data, save_path, verbose, save_outputs)
        #
        #             if verbose:
        #                 print('Time mark', name, '3', datetime.now() - prev_time_2)
        #
        #             if i != 0:
        #                 well_names &= set(predictions[name + '_orig'].columns)
        #             else:
        #                 well_names = set(predictions[name + '_orig'].columns)
        #             if verbose:
        #                 print(name, 'done')
        #
        #     if verbose:
        #         print('Time mark 2', datetime.now() - prev_time)
        #
        #     if flags.get('wc_flag', True) and flags.get('liquid_flag', True):
        #         predictions['oil_orig'] = \
        #             pd.DataFrame(
        #                 data=[(1 - predictions['wc_orig'][well_name]) * predictions['liquid_orig'][well_name] for
        #                       well_name
        #                       in well_names],
        #                 columns=pd.date_range(start=PREDICTION_DATE, end=PREDICTION_DATE2),
        #                 index=list(well_names)).T
        #
        #         predictions['oil_cum'] = pd.DataFrame(
        #             np.cumsum(predictions['oil_orig'].values, axis=0),
        #             index=predictions['oil_orig'].index,
        #             columns=predictions['oil_orig'].columns)
        #
        #         if verbose:
        #             print('Time mark 3', datetime.now() - prev_time)
        #
        #         save_path_orig = os.path.join(os.path.join(save_path, 'oil'), 'orig')
        #         save_path_cum = os.path.join(os.path.join(save_path, 'oil'), 'cum')
        #         if not os.path.exists(save_path_orig):
        #             os.makedirs(save_path_orig)
        #         if not os.path.exists(save_path_cum):
        #             os.makedirs(save_path_cum)
        #
        #         integr_pred = []
        #         pred_list = []
        #
        #         for well_name, df in predictions['oil_orig'].iteritems():
        #             integr_pred.append(df.sum(axis=0))
        #             pred_list.append(df)
        #
        #             if verbose:
        #                 # Get plots predicted vs real for every well
        #                 plt.figure(figsize=(16, 5))
        #                 date_range = pd.date_range(start=PREDICTION_DATE,
        #                                            end=PREDICTION_DATE2)
        #                 plt.plot(date_range, df, 'bo-', label='Прогноз')
        #                 plt.legend(loc='upper left')
        #                 # plt.ylabel('Дебит жидкости')
        #                 plt.title('Прогноз на скважине номер ' + well_name)
        #                 plt.grid(True)
        #                 plt.savefig(os.path.join(save_path_orig, '{}.png'.format(well_name)))
        #                 plt.close()
        #
        #         pred_df = pd.concat(pred_list, axis=1)
        #         pred_df.index = pd.date_range(start=PREDICTION_DATE,
        #                                       end=PREDICTION_DATE2)
        #
        #         if save_outputs:
        #             tmp_ind = [name for name in predictions['oil_orig'].columns]
        #             pd.DataFrame(np.array(integr_pred).flatten(), index=tmp_ind).to_excel(
        #                 os.path.join(save_path_orig, '{}_INTEGR_PRED_ORIG.xlsx'.format('oil')), header=False)
        #
        #             pred_df.to_excel(
        #                 os.path.join(save_path_orig, '{}_ORIG.xlsx'.format('oil')))
        #
        #         if verbose:
        #             print('Time mark 4', datetime.now() - prev_time)
        #
        #         # CUM
        #         integr_pred = []
        #         pred_list = []
        #
        #         for well_name, df in predictions['oil_cum'].iteritems():
        #             integr_pred.append(df.sum(axis=0))
        #             pred_list.append(df)
        #
        #             if verbose:
        #                 # Get plots predicted vs real for every well
        #                 plt.figure(figsize=(16, 5))
        #                 date_range = pd.date_range(start=PREDICTION_DATE,
        #                                            end=PREDICTION_DATE2)
        #                 plt.plot(date_range, df, 'bo-', label='Прогноз')
        #                 plt.legend(loc='upper left')
        #                 # plt.ylabel('Дебит жидкости')
        #                 plt.title('Прогноз на скважине номер ' + well_name)
        #                 plt.grid(True)
        #                 plt.savefig(os.path.join(save_path_cum, '{}.png'.format(well_name)))
        #                 plt.close()
        #
        #         if save_outputs:
        #             pred_df = pd.concat(pred_list, axis=1)
        #             pred_df.index = pd.date_range(start=PREDICTION_DATE,
        #                                           end=PREDICTION_DATE2)
        #
        #             tmp_ind = [name for name in predictions['oil_cum'].columns]
        #             pd.DataFrame(np.array(integr_pred).flatten(), index=tmp_ind).to_excel(
        #                 os.path.join(save_path_cum, '{}_INTEGR_PRED_CUM.xlsx'.format('oil')), header=False)
        #
        #             pred_df.to_excel(
        #                 os.path.join(save_path_cum, '{}_CUM.xlsx'.format('oil')))
        #         if verbose:
        #             print('Time mark 4', datetime.now() - prev_time)
        #
        #     return predictions, pressure, injection

    @staticmethod
    def _get_info(injection, pressure):
        info = {
            'name': 'hybrid',
            'version': '0.0.1',
            'restart_date': datetime.strftime(injection.index[-1] + timedelta(1), '%Y-%m-%d'),
            'prod_wells': {},
            'inj_wells': {}
        }
        for (well_name, inj) in injection.items():
            info['inj_wells'][well_name] = {'injection': inj[-1]}

        for (well_name, press) in pressure.items():
            info['prod_wells'][well_name] = {'pressure': press[-1]}

        return info

    @property
    def info(self):
        return self._info

    @property
    def config(self):
        return self._config_from_train

    @property
    def CRM(self):
        return self._CRM

    @property
    def CRMIP(self):
        return self._CRMIP

    @property
    def pred_CRM(self):
        return self._pred_CRM

    @property
    def pred_CRMIP(self):
        return self._pred_CRMIP

    @property
    def f(self):
        return self._f

    @property
    def tau(self):
        return self._tau

    @property
    def J(self):
        return self._J
