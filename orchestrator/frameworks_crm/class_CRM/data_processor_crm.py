from pathlib import Path

import pandas as pd

from frameworks_crm.class_CRM.config import ConfigCRM


class DataProcessorCRM:
    # Columns to read
    _prod_columns = [
        'well.ois',
        'dt',
        # 'Давление забойное от Hд',
        # 'Давление забойное от Pпр',
        'Давление забойное',
        'Дебит жидкости среднесуточный',
        'Дебит нефти расчетный',
        'Дебит газа (ТМ)',
        # 'Обводненность объемная',
        # 'Обводненность (ТМ)',
        # 'Обводненность (ХАЛ)',
        'charwork.name',
        'sost',
    ]

    _inj_columns = [
        'well.ois',
        'dt',
        # 'Приемистость ТМ',
        # 'Приемистость исходная по ТМ',
        # 'Приемистость сервисных служб',
        'Приемистость среднесуточная',
        'charwork.name',
        'sost',
    ]

    def __init__(self, config: ConfigCRM):
        self._config = config
        self._path = self._get_path(config.oilfield)
        self._run()

    def _run(self):
        self._read_data()
        self._drop_unnecessary_dates()
        self._initialize_data()
        self._well_names = self._get_well_names()
        self._get_required_data_format()
        self._set_column_names()
        self._parse_well_names()
        self._set_indexes()
        self._interpolate_governments()
        self._save_input_data_CRM(self._path, self._config.oilfield)

    def _read_data(self):
        self._fond = pd.read_feather(self._path / 'fond.feather')
        self._frac = pd.read_feather(self._path / 'frac.feather')
        self._coord = pd.read_feather(self._path / 'coord.feather')
        self._wellplast = pd.read_feather(self._path / 'wellplast.feather')
        self._welllist = pd.read_feather(self._path / 'welllist.feather')
        self._projectcoord = pd.read_feather(self._path / 'projectcoord.feather')
        columns = list(set(self._prod_columns + self._inj_columns))
        self._sh_sost_fond = pd.read_feather(self._path / 'sh_sost_fond.feather', columns=columns)

    def _drop_unnecessary_dates(self):
        self._sh_sost_fond = self._sh_sost_fond[
            (self._sh_sost_fond.dt >= self._config.date_start_adapt) &
            (self._sh_sost_fond.dt <= self._config.date_end_forecast)
            ]
        self._sh_sost_fond = self._sh_sost_fond[self._sh_sost_fond.sost == 'В работе']

    def _initialize_data(self):
        self._prod_wells = pd.DataFrame(columns=self._prod_columns)
        self._inj_wells = pd.DataFrame(columns=self._inj_columns)
        dates_test = pd.date_range(self._config.date_end_adapt, self._config.date_end_forecast, freq='D').date
        self._governm_prod = pd.DataFrame(index=dates_test)
        self._governm_inj = pd.DataFrame(index=dates_test)
        self._coordinates = pd.DataFrame(columns=['Координата X', 'Координата Y'])
        # TODO: раскомментить, когда будут данные по хвосту/пятке скважины
        # self._coordinates = pd.DataFrame(columns=['Координата X', 'Координата Y',
        #                                           'Координата X_1', 'Координата Y_1'])

    def _get_well_names(self):
        well_names = self._sh_sost_fond['well.ois'].unique()
        return well_names

    def _get_required_data_format(self):
        for name in self._well_names:
            well_fond = self._fond[self._fond['well.ois'] == name]
            # Выделение ПОСТОЯННОГО режима на прогнозе, иначе - скважина не рассматривается
            after_test = well_fond.dtstart < self._config.date_end_adapt
            before_end = well_fond.dtend > self._config.date_end_forecast
            regime = well_fond[after_test & before_end]
            if regime.empty:
                continue

            regime = regime.iloc[0]
            dtstart = regime['dtstart']  # Дата начала последнего режима скважины
            well_chess = self._sh_sost_fond[self._sh_sost_fond['well.ois'] == name]
            well_chess = well_chess.loc[well_chess.dt >= dtstart]

            if regime['charwork.name'] == 'Нефтяные':
                well_chess = well_chess[self._prod_columns]
                watercut = 1 - well_chess['Дебит нефти расчетный'] / well_chess['Дебит жидкости среднесуточный']
                well_chess['Обводненность'] = watercut
                self._prod_wells = pd.concat([self._prod_wells, well_chess], sort=False)

                well_chess.set_index('dt', inplace=True)
                self._governm_prod[name] = well_chess['Дебит жидкости среднесуточный']
                self._governm_prod[name] = well_chess['Дебит газа (ТМ)']
            elif regime['charwork.name'] == 'Нагнетательные':
                well_chess = well_chess[self._inj_columns]
                self._inj_wells = pd.concat([self._inj_wells, well_chess])

                well_chess.set_index('dt', inplace=True)
                self._governm_inj[name] = well_chess['Приемистость среднесуточная']

            self._formation_names = self._get_formation_names(well_name_ois=name)
            well_coords = self._get_coords(well_name_ois=name)
            if not well_coords.isna().values.any():
                self._coordinates = pd.concat([self._coordinates, well_coords])
            else:
                print(f'Well {name} is ignored. No info on coordinates!')

    def _get_formation_names(self, well_name_ois) -> list:
        """Устанавливает OIS названия пластов скважины.

        Notes:
            Определяются названия пластов, которые имели хоть одну активную дату
            в пределах date_start_adapt и date_test.
        """
        # TODO: Понять, надо ли сдвигать date_start_adapt для случая, когда скважина имеет несколько пластов.
        #  Пример 1: к скважине последовательно подключаются 2 пласта до date_test.
        #  Пример 2: скважина работает на 2 пластах, один пласт отключается до date_test.
        df = self._wellplast.loc[
            (self._wellplast['well.ois'] == well_name_ois) &
            (self._wellplast['dtstart'] < self._config.date_end_adapt) &
            (self._wellplast['dtend'] >= self._config.date_start_adapt)
            ]
        formation_names = df['plast'].to_list()
        return formation_names

    def _set_column_names(self):
        self._prod_wells.columns = ['Скважина',
                                    'Дата',
                                    'Забойное давление',
                                    'Дебит жидкости',
                                    'Дебит нефти',
                                    'Дебит газа',
                                    'Тип скважины',
                                    'Состояние',
                                    'Обводненность', ]
        self._inj_wells.columns = ['Скважина', 'Дата', 'Приемистость', 'Тип скважины', 'Состояние']

    def _parse_well_names(self):
        # Переименовываем название скважин
        prod_well_name_list = list(set(self._prod_wells.iloc[:, 0]))
        inj_well_name_list = list(set(self._inj_wells.iloc[:, 0]))
        rename_prod_well_name_list = [self._parse_well_name(name_well) for name_well in prod_well_name_list]
        rename_inj_well_name_list = [self._parse_well_name(name_well) for name_well in inj_well_name_list]
        self._prod_wells = self._prod_wells.replace(prod_well_name_list, rename_prod_well_name_list)
        self._inj_wells = self._inj_wells.replace(inj_well_name_list, rename_inj_well_name_list)

    def _set_indexes(self):
        self._prod_wells.set_index('Скважина', inplace=True)
        self._inj_wells.set_index('Скважина', inplace=True)
        self._coordinates.index.name = 'Скважина'

    def _interpolate_governments(self):
        self._governm_prod.ffill(inplace=True)
        self._governm_prod.bfill(inplace=True)
        self._governm_inj.ffill(inplace=True)
        self._governm_inj.bfill(inplace=True)

    def _get_coords(self, well_name_ois) -> pd.DataFrame:
        df_projectcoord = self._projectcoord
        df_coord = self._coord.loc[self._coord['well.ois'] == well_name_ois]
        layer_names = df_projectcoord.loc[df_projectcoord['idproject'].isin(self._formation_names)]['id'].to_list()
        # TODO: Найти причину отсутствия layer_names.
        #  Пример: Крайнее, 256000610, пласт БС9/1 был записан как БС9%2F1.
        if not layer_names:
            x_coord = df_coord['x'].mean()
            y_coord = df_coord['y'].mean()
        else:
            x_coord = df_coord.loc[df_coord['plastcoord'].isin(layer_names)]['x'].mean()
            y_coord = df_coord.loc[df_coord['plastcoord'].isin(layer_names)]['y'].mean()
        print(well_name_ois, f'x={x_coord}', f'y={y_coord}', sep='\t')
        df_coord = pd.DataFrame(columns=['Координата X', 'Координата Y'], index=[well_name_ois])
        # TODO: искать координаты носка и пятки скважины для новой версии CRM
        # df_coord = pd.DataFrame(columns=['Координата X', 'Координата Y', 'Координата X_1', 'Координата Y_1'],
        #                         index=[well_name_ois])
        df_coord[['Координата X', 'Координата Y']] = [x_coord, y_coord]
        df_coord = df_coord.rename(index={well_name_ois: str(self._parse_well_name(well_name_ois))})
        return df_coord

    def _parse_well_name(self, name_ois):
        well_name = self._welllist[self._welllist.ois == name_ois]
        well_name = well_name[well_name.npath == 0]
        well_name = well_name.at[well_name.index[0], 'num']
        return str(well_name)

    def _save_input_data_CRM(self, path, oilfield):
        with pd.ExcelWriter(f'{path}/Input_data_CRM_{oilfield}.xlsx') as writer:
            self._prod_wells.to_excel(writer, sheet_name='Добывающие')
            self._inj_wells.to_excel(writer, sheet_name='Нагнетательные')
            self._coordinates.to_excel(writer, sheet_name='Координаты')

    @staticmethod
    def _get_path(oilfield):
        project_root = Path(__file__).parent.parent.parent
        return project_root / 'tools_preprocessor' / 'data' / f'{oilfield}'

    @property
    def prod_wells(self):
        return self._prod_wells

    @property
    def inj_wells(self):
        return self._inj_wells

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def governm_prod(self):
        return self._governm_prod

    @property
    def governm_inj(self):
        return self._governm_inj
