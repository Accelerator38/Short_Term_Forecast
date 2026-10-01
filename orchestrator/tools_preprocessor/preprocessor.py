import datetime
import json
import pathlib
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from loguru import logger

from frameworks_ftor.ftor.well import Well as WellFtor
from frameworks_ftor.ftor import wrapper_estimator
from frameworks_wolfram.wolfram.well import Well as WellWolfram
from tools_preprocessor.config import Config

DEFAULT_WELL_KIND = 'Горизонтально'
DEFAULT_FRAC_DATE_MIN = datetime.date(2000, 1, 1)
DEFAULT_FRAC_DATE_MAX = datetime.date(2100, 1, 1)  # Дата индикатор отсутствия мероприятия ГРП
DEFAULT_THICKNESS = 3
DEFAULT_THICKNESS_MAX = 50
DEFAULT_POROSITY = 0.2
DEFAULT_SATURATION_OIL = 0.5
DEFAULT_COMPRESSIBILITY_ROCK = 0.00005
DEFAULT_COMPRESSIBILITY_OIL = 0.0001
DEFAULT_COMPRESSIBILITY_WATER = 0.00004
DEFAULT_DENSITY_OIL = 0.85
DEFAULT_VISCOSITY_LIQ = 1.5
DEFAULT_VOLUME_FACTOR_LIQ = 1.15


class Preprocessor(object):

    _path_general = pathlib.Path(__file__).resolve().parent / 'data'
    _tables = [
        'coord',
        'fond',
        'frac',
        'merop',
        'mersum',
        'projectcoord',
        'sh_sost_fond',
        'sost',
        'sppl',
        'troil',
        'welllist',
        'wellplast',
    ]
    _fonds = [
        'Нефтяные',
    ]
    _sosts = [
        'В работе',
    ]

    def __init__(
            self,
            config: Config,
    ):
        self._config = config
        self._run()

    def create_wells_ftor(
            self,
            well_names_desire: list[int],
            user_constraints_for_adap_period: None or dict[str, float or list[float]] = None
    ) -> list[WellFtor]:
        wells = []
        well_names_exist = sorted(set(self._well_names) & set(well_names_desire))
        if not well_names_exist:
            raise AssertionError(
                'Список скважин well_names_desire не содержит ни одной скважины, доступной для расчета согласно '
                'заданным date_start, date_test, date_end. Проверьте список.'
            )
        for well_name in well_names_exist:
            try:
                creator_well = _CreatorWellFtor(
                    self._data,
                    self._config,
                    well_name,
                    user_constraints_for_adap_period,
                )
                wells.append(creator_well.well)
                print(f'well {well_name} was prepared\n')
            except Exception as exc:
                logger.exception(f'Preprocessor: FAIL {well_name}', exc)
        return wells

    def create_wells_wolfram(self, well_names_desire: list[int]) -> list[WellWolfram]:
        wells = []
        well_names_exist = sorted(set(self._well_names) & set(well_names_desire))
        if not well_names_exist:
            raise AssertionError(
                'Список скважин well_names_desire не содержит ни одной скважины, доступной для расчета согласно '
                'заданным date_start, date_test, date_end. Проверьте список.'
            )
        for well_name in well_names_exist:
            creator_well = _CreatorWellWolfram(
                self._data,
                self._config,
                well_name,
            )
            wells.append(creator_well.well)
        return wells

    def _run(self) -> None:
        self._check_dir_existence()
        self._read_data()
        self._handle_data()
        self._select_well_names()

    def _check_dir_existence(self) -> None:
        self._path_current = self._path_general / self._config.field_name
        if not self._path_current.exists():
            raise FileNotFoundError(f'Директория "{self._path_current}" не существует.')

    def _read_data(self) -> None:
        self._data = {}
        for table in self._tables:
            self._data[table] = pd.read_feather(self._path_current / f'{table}.feather')
        self._read_gdis_from_xlsm()

        df_constrs = pd.read_excel(self._path_general / 'constraint_settings.xlsx', index_col='field_name')
        df_events = pd.read_excel(self._path_general / 'event_settings.xlsx', index_col='event_name')
        self._data['constr_settings'] = df_constrs
        self._data['event_settings'] = df_events

    def _read_gdis_from_xlsm(self) -> None:
        cols_types = {
            'Скважина': str,
            'Тип скважины': str,
            'Пласт ОИС': str,
            'Дата окончания исследования': str,
            'Кпр, мД': str,
            'Кгидр, Д*см/сПз': float,
            'Lэфф,м': float,
            'Xf': float,
            'Цвет Кпр, мД': int,  # К - кириллическая
            'Цвет Lэфф,м': int,
            'Цвет Xf': int,
        }

        df = pd.read_excel(
            io=self._path_current / 'gdis.xlsm',
            usecols=cols_types.keys(),
            dtype=cols_types,
        )
        df = df.loc[df['Тип скважины'].isin([
            'доб',
            'добыв.',
            'Добывающая',
            'Фонт.',
            'мех.фонд',
        ])]
        df.dropna(subset=['Скважина', 'Дата окончания исследования', 'Пласт ОИС'], inplace=True)
        df['Дата окончания исследования'] = df['Дата окончания исследования'].apply(
            lambda date: datetime.datetime.strptime(date, '%d.%m.%Y').date()
        )
        df['Пласт ОИС'] = df['Пласт ОИС'].apply(lambda string: set(string.split()))
        self._data['gdis'] = df

    def _handle_data(self) -> None:
        self._handle_sppl()
        self._handle_troil()

    def _handle_sppl(self) -> None:
        self._data['sppl'].replace(
            to_replace={
                'pm': {0: DEFAULT_POROSITY},
                'nb': {0: DEFAULT_SATURATION_OIL},
                'sp': {0: DEFAULT_COMPRESSIBILITY_ROCK},
                'sn': {0: DEFAULT_COMPRESSIBILITY_OIL},
                'sw': {0: DEFAULT_COMPRESSIBILITY_WATER},
                'hs': {0: DEFAULT_DENSITY_OIL},
            },
            inplace=True,
        )

    def _handle_troil(self) -> None:
        self._data['troil'].replace(
            to_replace={
                'mu_liq': {0: DEFAULT_VISCOSITY_LIQ},
                'ob_kt': {0: DEFAULT_VOLUME_FACTOR_LIQ},
                'oilsaturatedthickness': {0: DEFAULT_THICKNESS},
            },
            inplace=True,
        )
        # Если в таблице столбец grpdate содержит дату, меньшую допустимой, то предполагается,
        # что данная дата не является показательной для расчета и заменяется на максимально возможную.
        self._data['troil']['grpdate'].where(
            self._data['troil']['grpdate'] > DEFAULT_FRAC_DATE_MIN, other=DEFAULT_FRAC_DATE_MAX, inplace=True)
        self._handle_troil_by_skvtype()

    def _handle_troil_by_skvtype(self) -> None:
        df = self._data['troil']
        kind_names_wrong = [
            '',
            'Прочие',
        ]
        well_names = df.loc[df['skvtype'].isin(kind_names_wrong)]['well.ois'].unique()
        for well_name in well_names:
            df_well = df.loc[df['well.ois'] == well_name].copy()
            df_well_correct_types = df_well.loc[~df_well['skvtype'].isin(kind_names_wrong)]
            if df_well_correct_types.empty:
                df_well['skvtype'] = DEFAULT_WELL_KIND
            else:
                kind_name = df_well_correct_types['skvtype'].value_counts().idxmax()
                df_well['skvtype'] = kind_name
            self._data['troil'].update(df_well)

    def _select_well_names(self) -> None:
        df_train = self._data['sh_sost_fond'].loc[
            (self._data['sh_sost_fond']['dt'] >= self._config.date_start) &
            (self._data['sh_sost_fond']['dt'] < self._config.date_test)
        ]
        df_test = self._data['sh_sost_fond'].loc[
            (self._data['sh_sost_fond']['dt'] >= self._config.date_test) &
            (self._data['sh_sost_fond']['dt'] <= self._config.date_end)
        ]
        names_train = self._select_well_names_unique(df_train)
        names_test = self._select_well_names_unique(df_test)
        names_by_shops = self._select_well_names_unique_by_ceh()
        self._well_names = sorted(set(names_train) & set(names_test) & set(names_by_shops))

    def _select_well_names_unique(self, df: pd.DataFrame) -> list[int]:
        df = df.loc[
            (df['charwork.name'].isin(self._fonds)) &
            (df['sost'].isin(self._sosts))
        ]
        well_names = df['well.ois'].unique().tolist()
        return well_names

    def _select_well_names_unique_by_ceh(self) -> list[int]:
        df = self._data['welllist'].loc[self._data['welllist']['ceh'].isin(self._config.shops)]
        well_names = df['ois'].unique().tolist()
        return well_names

    @property
    def path(self) -> pathlib.Path:
        return self._path_current

    @property
    def well_names(self) -> list[int]:
        return self._well_names


class _CreatorWell(ABC):

    _NAME_RATE_LIQ = 'Дебит жидкости среднесуточный'
    _NAME_RATE_OIL = 'Дебит нефти расчетный'
    _NAME_CUM_LIQ = 'Добыча жидкости накопленная'
    _NAME_CUM_OIL = 'Добыча нефти накопленная'
    _NAME_PRESSURE = 'Давление забойное'
    _NAME_WATERCUT = 'Обводненность объемная'
    _NAME_RATE_GAS = 'Дебит газа (ТМ)'

    def __init__(
            self,
            data: dict[str, pd.DataFrame],
            config: Config,
            well_name_ois: int,
    ):
        self._data = data
        self._field_name = config.field_name
        self._date_start = config.date_start
        self._date_test = config.date_test
        self._date_end = config.date_end
        self._well_name_ois = well_name_ois

    @abstractmethod
    def _run(self) -> None:
        pass

    @abstractmethod
    def _set_chess(self) -> None:
        df = self._data['sh_sost_fond'].loc[
            (self._data['sh_sost_fond']['well.ois'] == self._well_name_ois)
        ]
        self._df_chess = df.copy()
        self._df_chess.drop(columns='well.ois', inplace=True)
        self._df_chess.set_index(keys='dt', inplace=True, verify_integrity=True)
        self._df_chess.sort_index(inplace=True)
        self._df_chess[self._NAME_RATE_LIQ].loc[self._df_chess['sost'] == 'Остановлена'] = 0
        self._df_chess[self._NAME_RATE_OIL].loc[self._df_chess['sost'] == 'Остановлена'] = 0

    @abstractmethod
    def _set_well(self) -> None:
        pass

    @property
    @abstractmethod
    def well(self) -> WellFtor or WellWolfram:
        pass


class _CreatorWellFtor(_CreatorWell):

    _NAME_START_ADAP = 'Начало адаптации'

    _kind_codes_frac_no = {
        'Вертикально': 0,
        'Наклонно-направленно': 0,
        'Горизонтально': 1,
    }
    _kind_codes_frac = {
        'Вертикально': 2,
        'Наклонно-направленно': 2,
        'Горизонтально': 3
    }
    _prms_poss_for_constraints = {0: (
        'kind_code',
        'permeability',
        'skin',
        'res_radius',
        'pressure_initial',
    )}
    _prms_poss_for_constraints[1] = _prms_poss_for_constraints[0] + ('length_hor_well_bore',)
    _prms_poss_for_constraints[2] = _prms_poss_for_constraints[0] + ('length_half_fracture',)
    _prms_poss_for_constraints[3] = _prms_poss_for_constraints[0] + ('length_hor_well_bore', 'length_half_fracture')

    _prms_try_to_improve_bounds = {'permeability', 'pressure_initial', 'length_hor_well_bore', 'length_half_fracture'}

    def __init__(
            self,
            data: dict[str, pd.DataFrame],
            config: Config,
            well_name_ois: int,
            user_constraints_for_adap_period: None or dict[str, float or list[float]],
    ):
        super().__init__(
            data,
            config,
            well_name_ois,
        )
        self._user_constrs = user_constraints_for_adap_period
        self._run()

    def _run(self) -> None:
        self._set_well_name_geo()
        self._set_formation_names()
        self._set_formation_properties_from_sppl()
        self._set_formation_properties_from_troil()
        self._set_formation_cross_coords()
        self._set_chess()
        self._set_flood()
        self._set_well()

    def _set_well_name_geo(self) -> None:
        """Устанавливает ГеоБД номер скважины.

        Notes:
            Номер скважины определяется как ГеоБД номер последнего вводимого
            ствола скважины. Данный ствол должен быть активным на момент
            date_test.
        """
        # TODO: Понять, надо ли сдвигать date_start для случая, когда скважина имеет несколько стволов.
        #  Пример 1: на скважине появляется новый ствол в период с date_start до date_test.
        #  Пример 2: на скважине есть 2 активных ствола, в период с date_start до date_test один из них отключается.
        df = self._data['welllist'].loc[
            (self._data['welllist']['ois'] == self._well_name_ois) &
            (self._data['welllist']['dtstart'] < self._date_test) &
            (self._data['welllist']['dtend'] >= self._date_test)
        ]
        date_start_max = df['dtstart'].max()
        self._well_name_geo = df.loc[df['dtstart'] == date_start_max]['well'].iloc[-1]

    def _set_formation_names(self) -> None:
        """Устанавливает OIS названия пластов скважины.

        Notes:
            Определяются названия пластов, которые имели хоть одну активную дату
            в пределах date_start и date_test.
        """
        # TODO: Понять, надо ли сдвигать date_start для случая, когда скважина имеет несколько пластов.
        #  Пример 1: к скважине последовательно подключаются 2 пласта до date_test.
        #  Пример 2: скважина работает на 2 пластах, один пласт отключается до date_test.
        df = self._data['wellplast'].loc[
            (self._data['wellplast']['well.ois'] == self._well_name_ois) &
            (self._data['wellplast']['dtstart'] < self._date_test) &
            (self._data['wellplast']['dtend'] >= self._date_start)
        ]
        self._formation_names = df['plast'].to_list()

    def _set_formation_properties_from_sppl(self) -> None:
        df = self._data['sppl'].loc[
            (self._data['sppl']['plastmer'].isin(self._formation_names)) &
            (self._data['sppl']['tk'] < self._date_test)
        ]
        self._porosity = df['pm'].mean()
        self._c_r = df['sp'].mean()
        self._c_w = df['sw'].mean()
        self._c_o = df['sn'].mean()
        self._s_o = df['nb'].mean()
        self._compressibility_total = (self._c_o * self._s_o + self._c_w * (1 - self._s_o) + self._c_r) * 10
        self._density_oil = df['hs'].mean()

    def _set_formation_properties_from_troil(self) -> None:
        df = self._data['troil'].loc[
            (self._data['troil']['well.ois'] == self._well_name_ois) &
            (self._data['troil']['plastmer'].isin(self._formation_names)) &
            (self._data['troil']['dt'] >= self._date_start) &
            (self._data['troil']['dt'] < self._date_test)
        ]
        if not df.empty:
            kind_name = df['skvtype'].value_counts().idxmax()
            frac_date = df['grpdate'].iloc[0]
            if frac_date <= self._date_start:
                self._kind_code = self._kind_codes_frac[kind_name]
            else:
                self._kind_code = self._kind_codes_frac_no[kind_name]
            df.set_index(keys=['plastmer', df.index], inplace=True, verify_integrity=True)
            self._thickness = df['oilsaturatedthickness'].mean(level='plastmer').sum()
            if self._thickness >= DEFAULT_THICKNESS_MAX:
                self._thickness = DEFAULT_THICKNESS
            self._viscosity_liq = df['mu_liq'].mean()
            self._volume_factor_liq = df['ob_kt'].mean()
        else:
            self._kind_code = self._kind_codes_frac_no[DEFAULT_WELL_KIND]
            self._thickness = DEFAULT_THICKNESS
            self._viscosity_liq = DEFAULT_VISCOSITY_LIQ
            self._volume_factor_liq = DEFAULT_VOLUME_FACTOR_LIQ

    def _set_formation_cross_coords(self) -> None:
        df_projectcoord = self._data['projectcoord']
        df_coord = self._data['coord'].loc[self._data['coord']['well.ois'] == self._well_name_ois]
        layer_names = df_projectcoord.loc[df_projectcoord['idproject'].isin(self._formation_names)]['id'].to_list()
        # TODO: Найти причину отсутствия layer_names.
        #  Пример: Крайнее, 256000610, пласт БС9/1 был записан как БС9%2F1.
        if not layer_names:
            self._x_coord = df_coord['x'].mean()
            self._y_coord = df_coord['y'].mean()
        else:
            self._x_coord = df_coord.loc[df_coord['plastcoord'].isin(layer_names)]['x'].mean()
            self._y_coord = df_coord.loc[df_coord['plastcoord'].isin(layer_names)]['y'].mean()
        print(self._well_name_geo, f'x={self._x_coord}', f'x={self._y_coord}', sep='\t')

    def _set_chess(self) -> None:
        super()._set_chess()
        self._df_chess = self._df_chess.loc[self._date_start:self._date_end]
        self._df_chess = self._df_chess.loc[self._df_chess['charwork.name'] == 'Нефтяные']
        self._date_start = self._df_chess.index[0]

        if np.any(self._df_chess.index != pd.date_range(self._date_start, self._date_end, freq="D")):
            raise IndexError(f'There is fragmentation of input data')
        elif any(self._df_chess[self._NAME_PRESSURE].loc[self._df_chess['sost'] == 'В работе'].isna()):
            raise ValueError(f'There are missing values in input data for column "{self._NAME_PRESSURE}"')

        self._df_chess[self._NAME_RATE_OIL] = self._df_chess[self._NAME_RATE_OIL].apply(
            lambda x: x / self._density_oil)
        rates_liq = self._df_chess[self._NAME_RATE_LIQ]
        rates_oil = self._df_chess[self._NAME_RATE_OIL]
        self._df_chess[self._NAME_WATERCUT] = (rates_liq - rates_oil) / rates_liq
        self._set_chess_prm_constraints()

    def _set_chess_prm_constraints(self) -> None:
        events = self._df_chess['merid.name'].dropna()
        start_adap = pd.Series(data=[self._NAME_START_ADAP], index=[self._date_start])
        events = events.append(start_adap)
        for event_date, event_name in events.iteritems():
            if event_name == 'ГРП':
                if self._kind_code == 0:
                    self._kind_code = 2
                elif self._kind_code == 1:
                    self._kind_code = 3

            # TODO: У Фтора жесткое API в плане разбиения на периоды, нужно либо сделать
            #  его более гибким, либо продумать четкие правила обработки данных
            day = event_date - datetime.timedelta(days=1)
            work_yesterday = self._df_chess.loc[day, 'sost'] == 'В работе' if day in self._df_chess.index else False
            add_to_chess = self._df_chess.loc[event_date, 'sost'] == 'В работе' and not work_yesterday
            # TODO: event_name == self._NAME_START_ADAP - Костыль, нужно перенести обрезку df_chess
            if add_to_chess or event_name == self._NAME_START_ADAP:
                prm_constrs = self._get_prm_constraints(event_date, event_name)
                if len(prm_constrs) > 0:
                    self._df_chess.loc[event_date, 'prm_constraints'] = json.dumps(prm_constrs)

    def _set_flood(self) -> None:
        df = self._data['mersum'].loc[
            (self._data['mersum']['well.ois'] == self._well_name_ois) &
            (self._data['mersum']['plastmer'].isin(self._formation_names)) &
            (self._data['mersum']['dt'] < self._date_start)
        ]
        df_mersum = df.copy()
        df_mersum.drop(columns=['well.ois', 'plastmer'], inplace=True)
        df_mersum.dropna(axis=0, how='any', inplace=True)
        df_mersum.set_index(keys=['dt', df_mersum.index], inplace=True, verify_integrity=True)
        df_mersum = df_mersum.sum(axis=0, level='dt')
        prods_liq = df_mersum['liq']
        prods_oil = df_mersum['oilm3']
        df_mersum[self._NAME_WATERCUT] = (prods_liq - prods_oil) / prods_liq
        df_mersum.columns = [
            self._NAME_CUM_LIQ,
            self._NAME_CUM_OIL,
            self._NAME_WATERCUT,
        ]
        df_chess = self._df_chess.loc[
                   :self._date_test - datetime.timedelta(days=1),
                   [
                       self._NAME_RATE_LIQ,
                       self._NAME_RATE_OIL,
                       self._NAME_WATERCUT,
                   ]
                   ]
        df_chess.columns = df_mersum.columns
        self._df_flood = pd.concat(objs=[df_mersum, df_chess])
        self._df_flood.fillna(value=0, inplace=True)
        self._df_flood.update(self._df_flood[[self._NAME_CUM_LIQ, self._NAME_CUM_OIL]].cumsum())
        self._cum_liq_start = df_mersum[self._NAME_CUM_LIQ].sum()
        self._cum_liq_test = self._df_flood[self._NAME_CUM_LIQ].iloc[-1]
        self._cum_oil_test = self._df_flood[self._NAME_CUM_OIL].iloc[-1]

    def _set_well(self) -> None:
        df_chess = self._df_chess[[
            'sost',
            'merid.name',
            'prm_constraints',
            self._NAME_PRESSURE,
            self._NAME_WATERCUT,
            self._NAME_RATE_LIQ,
            self._NAME_RATE_OIL,
        ]]
        df_chess.columns = [
            WellFtor.NAME_STATUS,
            WellFtor.NAME_EVENT,
            WellFtor.NAME_CONSTRAINTS,
            WellFtor.NAME_PRESSURE,
            WellFtor.NAME_WATERCUT,
            WellFtor.NAME_RATE_LIQ,
            WellFtor.NAME_RATE_OIL,
        ]
        # TODO: Убрать установку date_start и обрезку df_chess в более подходящее место
        date_start = self._date_start
        start_work_date = df_chess[df_chess[WellFtor.NAME_STATUS] == 'В работе'].index[0]
        if start_work_date != date_start:
            prm_constrs = df_chess.loc[date_start, WellFtor.NAME_CONSTRAINTS]
            df_chess = df_chess[df_chess.index >= start_work_date]
            df_chess[WellFtor.NAME_CONSTRAINTS].iloc[0] = prm_constrs
            date_start = start_work_date

        df_flood = self._df_flood[[
            self._NAME_CUM_OIL,
            self._NAME_WATERCUT,
        ]]
        df_flood.columns = [
            WellFtor.NAME_CUM_OIL,
            WellFtor.NAME_WATERCUT,
        ]
        self._well = WellFtor(
            self._well_name_ois,
            date_start,
            self._date_test,
            self._date_end,
            self._thickness,
            self._porosity,
            self._compressibility_total,
            self._density_oil,
            self._viscosity_liq,
            self._volume_factor_liq,
            self._cum_liq_start,
            self._cum_liq_test,
            self._cum_oil_test,
            df_chess,
            self._df_flood,
        )
        # TODO: Внести новые аттрибуты в API Ftor.
        self._well.x_coord = self._x_coord
        self._well.y_coord = self._y_coord

    def _get_prm_constraints(self, event_date: datetime.date, event_name: str) -> dict[str, float or list[float]]:
        if event_name == self._NAME_START_ADAP:
            prms_for_constraints = self._prms_poss_for_constraints[self._kind_code]
        else:
            if event_name not in self._data['event_settings'].index:
                prms_for_constraints = ()
            else:
                prms_changed_by_event = json.loads(self._data['event_settings'].loc[event_name, 'changing params'])
                if event_date < self._date_test:
                    prms_for_constraints = tuple(set(self._prms_poss_for_constraints[self._kind_code]) &
                                                 set(prms_changed_by_event['adap_period']))
                else:
                    prms_for_constraints = tuple(set(self._prms_poss_for_constraints[self._kind_code]) &
                                                 set(prms_changed_by_event['test_period']))

        constraints = dict()
        prms_try_to_improve_bounds = self._prms_try_to_improve_bounds.copy()
        user_constrs_exist = self._user_constrs is not None
        for prm in prms_for_constraints:
            if user_constrs_exist and prm in self._user_constrs and event_date < self._date_test:
                prms_try_to_improve_bounds.discard(prm)
                constraints[prm] = self._user_constrs[prm]
            else:
                if prm == 'kind_code':
                    constraints[prm] = self._kind_code
                else:
                    use_field_name_row = False
                    if self._field_name in self._data['constr_settings'].index:
                        use_field_name_row = self._data['constr_settings'].loc[self._field_name, prm] is not np.nan
                    row = self._field_name if use_field_name_row else 'По умолчанию'
                    prm_bound_settings = json.loads(self._data['constr_settings'].loc[row, prm])
                    if event_date < self._date_test:
                        constraints[prm] = prm_bound_settings['bounds']
                    else:
                        constraints[prm] = prm_bound_settings['val_test_period']

        if event_date < self._date_test:
            #  Изменяет constraints
            _BoundsImprover(
                self._NAME_START_ADAP,
                prms_try_to_improve_bounds,
                constraints,
                event_name,
                event_date,
                self._data,
                self._date_test,
                self._date_end,
                self._field_name,
                self._well_name_ois,
                self._well_name_geo,
                self._NAME_PRESSURE,
                self._kind_code,
                self._formation_names,
                self._thickness,
                self._viscosity_liq,
            )
        return constraints

    @property
    def well(self) -> WellFtor:
        return self._well


class _CreatorWellWolfram(_CreatorWell):

    def __init__(
            self,
            data: dict[str, pd.DataFrame],
            config: Config,
            well_name_ois: int,
    ):
        super().__init__(
            data,
            config,
            well_name_ois,
        )
        self._run()

    def _run(self) -> None:
        self._set_chess()
        self._set_well()

    def _set_chess(self) -> None:
        super()._set_chess()
        self._df_chess = self._df_chess.loc[:self._date_end]

    def _set_well(self) -> None:
        if self._NAME_RATE_GAS not in self._df_chess.columns:
            self._df_chess[self._NAME_RATE_GAS] = 0.0

        df_chess = self._df_chess[[
            self._NAME_PRESSURE,
            self._NAME_RATE_LIQ,
            self._NAME_RATE_OIL,
            self._NAME_RATE_GAS,
        ]]
        df_chess.columns = [
            WellWolfram.NAME_PRESSURE,
            WellWolfram.NAME_RATE_LIQ,
            WellWolfram.NAME_RATE_OIL,
            WellWolfram.NAME_RATE_GAS,
        ]
        self._well = WellWolfram(
            self._well_name_ois,
            df_chess,
        )

    @property
    def well(self) -> WellWolfram:
        return self._well


class _BoundsImprover:
    _mark_code = {
        'not count': 16711680,
        'bad': 0,
        'good': 32768,
        'excellent': 255,
    }
    _NAME_K = wrapper_estimator.K
    _NAME_PI = wrapper_estimator.PI
    _NAME_XF = wrapper_estimator.XF
    _NAME_L_HOR = wrapper_estimator.L_HOR

    def __init__(
            self,
            start_adap_name: str,
            prms_try_to_improve_bounds: set[str],
            prm_constraints: dict[str, float or list[float]],
            event_name: str,
            date: datetime.date,
            data: dict[str, pd.DataFrame],
            date_test: datetime.date,
            date_end: datetime.date,
            field_name: str,
            well_name: int,
            well_name_geo: str,
            name_pressure,
            kind_code: int,
            formation_names: list[str],
            thickness: float,
            viscosity_liq: float,
    ):
        self._start_adap_name = start_adap_name
        self._prms_try_to_improve_bounds = prms_try_to_improve_bounds
        self._prm_constrs = prm_constraints
        self.event_name = event_name
        self._date = date
        self._data = data
        self._date_test = date_test
        self._date_end = date_end
        self._field_name = field_name
        self._well_name = well_name
        self._well_name_geo = well_name_geo
        self._name_pressure = name_pressure
        self._kind_code = kind_code
        self._formation_names = formation_names
        self._thickness = thickness
        self._viscosity_liq = viscosity_liq
        self._df_gdis = self._data['gdis'].copy()
        self._df_merop = self._data['merop'].loc[self._data['merop']['well.ois'] == self._well_name].copy()
        self._df_chess = self._data['sh_sost_fond'].loc[
            (self._data['sh_sost_fond']['well.ois'] == self._well_name)].copy()

        self._run()

    def _run(self):
        for prm in self._prms_try_to_improve_bounds:
            if prm in self._prm_constrs:
                is_prm_for_optimization = isinstance(self._prm_constrs[prm], list)
                if is_prm_for_optimization:
                    self._try_to_improve_bounds(prm)

    def _set_frac(self) -> None:
        self._df_frac = self._data['frac'].copy()
        self._df_frac.dropna(subset=['well.ois', 'frac_date'], inplace=True)
        self._df_frac = self._df_frac.loc[
            (self._df_frac['well.ois'] == self._well_name) &
            (self._df_frac['frac_date'] <= self._date)
            ]

    def _try_to_improve_k_bounds_well_plast(self) -> bool:
        cols = [
            'Скважина',
            'Дата окончания исследования',
            'Кпр, мД',
            'Кгидр, Д*см/сПз',
            'Цвет Кпр, мД',
            'Пласт ОИС',
        ]
        df = self._df_gdis[cols].copy()
        df.dropna(inplace=True)
        df = df.loc[df['Цвет Кпр, мД'] != self._mark_code['not count']]
        two_years = datetime.timedelta(2 * 365)
        past_bound_date = self._date - two_years
        future_bound_date = min(self._date + two_years, self._date_test)
        df = self._get_df_gdis_segment(
            df,
            past_bound_date,
            future_bound_date,
            well_filter=True,
            plast_filter=True,
        )
        if df.empty:
            return False
        df['Дата окончания исследования'] = df['Дата окончания исследования'].apply(self._get_abs_time_interval)
        df = df.loc[df['Дата окончания исследования'] == df['Дата окончания исследования'].min()]

        new_df = df.loc[df['Цвет Кпр, мД'] == self._mark_code['excellent']].copy()
        if not new_df.empty:
            k_gidr_init = new_df.iloc[0]['Кгидр, Д*см/сПз']
            k_init = 10 * self._viscosity_liq * k_gidr_init / self._thickness
            self._prm_constrs[self._NAME_K] = [0.7 * k_init, 1.3 * k_init]
            return True

        new_df = df.loc[df['Цвет Кпр, мД'] == self._mark_code['good']].copy()
        if not new_df.empty:
            k_gidr_init = new_df.iloc[0]['Кгидр, Д*см/сПз']
            k_init = 10 * self._viscosity_liq * k_gidr_init / self._thickness
            self._prm_constrs[self._NAME_K] = [0.3 * k_init, 1.7 * k_init]
            return True

        new_df = df.loc[df['Цвет Кпр, мД'] == self._mark_code['bad']].copy()
        if not new_df.empty:
            k_gidr_init = new_df.iloc[0]['Кгидр, Д*см/сПз']
            k_init = 10 * self._viscosity_liq * k_gidr_init / self._thickness
            self._prm_constrs[self._NAME_K] = [1 / 3 * k_init, 3 * k_init]
            return True

    def _try_to_improve_k_bounds_plast(self) -> bool:
        cols = [
            'Скважина',
            'Дата окончания исследования',
            'Кпр, мД',
            'Кгидр, Д*см/сПз',
            'Цвет Кпр, мД',
            'Пласт ОИС',
        ]
        df = self._df_gdis[cols].copy()
        df.dropna(inplace=True)
        df = df.loc[df['Цвет Кпр, мД'] != self._mark_code['not count']]
        df = self._get_df_gdis_segment(
            df,
            past_bound_date=None,
            future_bound_date=self._date_test,
            well_filter=False,
            plast_filter=True,
        )
        if df.empty:
            return False
        k_gidr_gdis_max = df['Кгидр, Д*см/сПз'].max()
        k_gidr_gdis_min = df['Кгидр, Д*см/сПз'].min()
        k_max = 10 * self._viscosity_liq * k_gidr_gdis_max / self._thickness * 1.1
        k_min = 10 * self._viscosity_liq * k_gidr_gdis_min / self._thickness / 1.1
        self._prm_constrs[self._NAME_K] = [k_min, k_max]
        return True

    def _try_to_improve_l_hor_bounds(self) -> bool:
        cols = [
            'Скважина',
            'Дата окончания исследования',
            'Lэфф,м',
            'Цвет Lэфф,м',
            'Пласт ОИС',
        ]
        df = self._df_gdis[cols].copy()
        df.dropna(inplace=True)
        df = df.loc[df['Цвет Lэфф,м'] != self._mark_code['not count']]
        two_years = datetime.timedelta(2 * 365)
        there_is_fracture = self._kind_code == 3
        if there_is_fracture:
            l_hor_gtms_list = ['ГРП']
            time_bounds_list = [self._date - two_years, self._date + two_years, self._date_test]
            past_bound_date, future_bound_date = self._get_bounds_dates_counting_gtms(l_hor_gtms_list, time_bounds_list)
        else:
            past_bound_date = self._date - two_years
            future_bound_date = min(self._date + two_years, self._date_test)
        df = self._get_df_gdis_segment(
            df,
            past_bound_date,
            future_bound_date,
            well_filter=True,
            plast_filter=True,
        )
        if df.empty:
            return False
        df['Дата окончания исследования'] = df['Дата окончания исследования'].apply(self._get_abs_time_interval)
        df = df.loc[df['Дата окончания исследования'] == df['Дата окончания исследования'].min()]

        new_df = df.loc[df['Цвет Lэфф,м'] == self._mark_code['excellent']].copy()
        if not new_df.empty:
            l_hor_init = new_df.iloc[0]['Lэфф,м']
            if there_is_fracture:
                self._prm_constrs[self._NAME_L_HOR] = [0.4 * l_hor_init, 1.6 * l_hor_init]
            else:
                self._prm_constrs[self._NAME_L_HOR] = [0.7 * l_hor_init, 1.3 * l_hor_init]
            return True

        new_df = df.loc[df['Цвет Lэфф,м'] == self._mark_code['good']].copy()
        if not new_df.empty:
            l_hor_init = new_df.iloc[0]['Lэфф,м']
            if there_is_fracture:
                self._prm_constrs[self._NAME_L_HOR] = [1 / 2 * l_hor_init, 2 * l_hor_init]
            else:
                self._prm_constrs[self._NAME_L_HOR] = [0.3 * l_hor_init, 1.7 * l_hor_init]
            return True

        new_df = df.loc[df['Цвет Lэфф,м'] == self._mark_code['bad']].copy()
        if not new_df.empty:
            l_hor_init = new_df.iloc[0]['Lэфф,м']
            self._prm_constrs[self._NAME_L_HOR] = [1 / 3 * l_hor_init, 3 * l_hor_init]
            return True

    def _try_to_improve_xf_bounds_gdis(self) -> bool:
        cols = [
            'Скважина',
            'Дата окончания исследования',
            'Xf',
            'Цвет Xf',
        ]
        df = self._df_gdis[cols].copy()
        df.dropna(inplace=True)
        df = df.loc[df['Цвет Xf'] != self._mark_code['not count']]
        xf_gtms_list = self._get_events_changing_xf()
        time_bounds_list = [self._date_test]
        past_bound_date, future_bound_date = self._get_bounds_dates_counting_gtms(xf_gtms_list, time_bounds_list)
        df = self._get_df_gdis_segment(
            df,
            past_bound_date,
            future_bound_date,
            well_filter=True,
            plast_filter=False,
        )
        if df.empty:
            return False
        df['Дата окончания исследования'] = df['Дата окончания исследования'].apply(self._get_abs_time_interval)
        df = df.loc[df['Дата окончания исследования'] == df['Дата окончания исследования'].min()]
        it_is_hor = self._kind_code == 3

        new_df = df.loc[df['Цвет Xf'] == self._mark_code['excellent']].copy()
        if not new_df.empty:
            xf_init = new_df.iloc[0]['Xf']
            if it_is_hor:
                self._prm_constrs[self._NAME_XF] = [0.4 * xf_init, 1.6 * xf_init]
            else:
                self._prm_constrs[self._NAME_XF] = [0.7 * xf_init, 1.3 * xf_init]
            return True

        new_df = df.loc[df['Цвет Xf'] == self._mark_code['good']].copy()
        if not new_df.empty:
            xf_init = new_df.iloc[0]['Xf']
            if it_is_hor:
                self._prm_constrs[self._NAME_XF] = [1 / 2 * xf_init, 2 * xf_init]
            else:
                self._prm_constrs[self._NAME_XF] = [0.3 * xf_init, 1.7 * xf_init]
            return True

        new_df = df.loc[df['Цвет Xf'] == self._mark_code['bad']].copy()
        if not new_df.empty:
            xf_init = new_df.iloc[0]['Xf']
            self._prm_constrs[self._NAME_XF] = [1 / 3 * xf_init, 3 * xf_init]
            return True

    def _try_to_improve_xf_bounds_frac(self) -> bool:
        df = self._df_frac.copy()
        df.dropna(subset=['xf'], inplace=True)
        df = df.loc[df['xf'] != 0]
        if df.empty:
            return False
        else:
            xf_frac = df.iloc[-1]['xf']
            self._prm_constrs[self._NAME_XF] = [1 / 5 * xf_frac, xf_frac]
            return True

    def _get_bounds_dates_counting_gtms(
            self,
            needed_gtms_list: list[str],
            time_bounds_list: list[datetime.date],
    ) -> tuple[datetime.date, datetime.date]:
        df = self._df_merop.loc[self._df_merop['merid.name'].isin(needed_gtms_list)]
        potential_bounds_dates = df['dtend'].tolist() + time_bounds_list
        potential_bounds_intervals = list(map(lambda date: (date - self._date).days, potential_bounds_dates))
        past_bound_index = self._get_negative_max_or_zero_index(potential_bounds_intervals)
        future_bound_index = self._get_positive_min_index(potential_bounds_intervals)
        if past_bound_index is None:
            past_bound_date = None
        else:
            past_bound_date = potential_bounds_dates[past_bound_index]
        if future_bound_index is None:
            future_bound_date = None
        else:
            future_bound_date = potential_bounds_dates[future_bound_index]
        return past_bound_date, future_bound_date

    def _get_df_gdis_segment(
            self,
            df_gdis: pd.DataFrame,
            past_bound_date: datetime.date or None,
            future_bound_date: datetime.date or None,
            well_filter: bool,
            plast_filter: bool,
    ) -> pd.DataFrame:
        if past_bound_date is not None:
            past_filter = past_bound_date < df_gdis['Дата окончания исследования']
        else:
            past_filter = True
        if future_bound_date is not None:
            future_filter = df_gdis['Дата окончания исследования'] <= future_bound_date
        else:
            future_filter = True
        if plast_filter is True:
            plast_filter = df_gdis['Пласт ОИС'] == set(self._formation_names)
        else:
            plast_filter = True
        if well_filter is True:
            well_filter = df_gdis['Скважина'] == self._well_name_geo
        else:
            well_filter = True
        df = df_gdis.loc[past_filter & future_filter & plast_filter & well_filter]
        return df

    def _get_abs_time_interval(self, table_date: datetime.date) -> int:
        return abs((table_date - self._date).days)

    @staticmethod
    def _get_negative_max_or_zero_index(array: list[int]) -> None or int:
        if len(array) == 0:
            return None
        i = 0
        while array[i] > 0:
            i += 1
            if i == len(array):
                return None
        neg_max = array[i]
        neg_max_index = i
        while i < len(array):
            if neg_max < array[i] <= 0:
                neg_max = array[i]
                neg_max_index = i
            i += 1
        return neg_max_index

    @staticmethod
    def _get_positive_min_index(array: list[int]) -> None or int:
        if len(array) == 0:
            return None
        i = 0
        while array[i] <= 0:
            i += 1
            if i == len(array):
                return None
        pos_min = array[i]
        pos_min_index = i
        while i < len(array):
            if 0 < array[i] < pos_min:
                pos_min = array[i]
                pos_min_index = i
            i += 1
        return pos_min_index

    def _try_to_improve_bounds(self, adap_prm: str) -> None:
        if adap_prm == self._NAME_K:
            if not self._try_to_improve_k_bounds_well_plast():
                self._try_to_improve_k_bounds_plast()

        elif adap_prm == self._NAME_PI:
            p_max = self._df_chess.loc[self._df_chess['dt'] < self._date_end][self._name_pressure].max()
            if p_max is not np.nan:
                self._prm_constrs[self._NAME_PI] = [p_max * 1.1, p_max * 3]

        elif adap_prm == self._NAME_L_HOR:
            self._try_to_improve_l_hor_bounds()

        elif adap_prm == self._NAME_XF:
            self._set_frac()
            if not self._try_to_improve_xf_bounds_gdis():
                self._try_to_improve_xf_bounds_frac()

    def _get_events_changing_xf(self) -> list[str]:
        events_changing_xf = []
        for event, row in self._data['event_settings'].iterrows():
            changing_prms_adap_prd = json.loads(row['changing params'])['adap_period']
            if self._NAME_XF in changing_prms_adap_prd:
                events_changing_xf.append(str(event))
        return events_changing_xf
