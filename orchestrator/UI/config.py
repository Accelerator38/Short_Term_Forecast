import datetime

from dateutil.relativedelta import relativedelta

FIELDS_SHOPS = {}
# FIELDS_SHOPS = {
#     'Валынтойское': ['ЦДНГ-12'],
#     'Вынгаяхинское': ['ЦДНГ-10'],
#     'Вынгапуровское': ['ЦДHГ-7', 'ЦДHГ-8'],  # H - латинская
#     'Крайнее': ['ЦДНГ-4', 'ЦДНГ-2'],
#     'Оренбургское': ['ЦДНГ-1'],
#     'Отдельное': ['ЦДHГ-1'],  # H - латинская
#     'Романовское': ['ЦДНГ-3'],
#     'Холмогорское': ['ЦДHГ-1'],  # H - латинская
#     'Восточно-Мессояхское': ['ЦДНГ-1'],
#     'Новогоднее': ['ЦДHГ-8'], # H - латинская
#     'им. Александра Жагрина': ['Зимнее'],
# }
FIELDS_SHOPS = dict(sorted(FIELDS_SHOPS.items()))

# Диапазон дат выгрузки sh таблицы
DATE_MIN = datetime.date(2000, 1, 1)
DATE_MAX = datetime.date(2025, 1, 1)

PERIOD_TRAIN_MIN = relativedelta(months=3)
PERIOD_TEST_MIN = relativedelta(months=1)

ML_FULL_ABBR = {
    'ElasticNet': 'ela',
    'LinearSVR': 'svr',
    'XGBoost': 'xgb',
}
YES_NO = {
    'Да': True,
    'Нет': False,
}

DEFAULT_FTOR_BOUNDS = {
    'permeability': {
        'label': 'Проницаемость k, мД',
        'lower_val': 0.1,
        'default_val': 5.,
        'upper_val': 10.,
        'step': 0.01,
        'min': 0.1,
        'max': 1000.,
        'help': '',
    },
    'skin': {
        'label': 'Skin',
        'lower_val': 0.1,
        'default_val': 1.,
        'upper_val': 2.,
        'step': 0.1,
        'min': 0.1,
        'max': 20.,
        'help': """Скин-фактор""",
    },
    'res_radius': {
        'label': 'Радиус резервуара, м',
        'lower_val': 200,
        'default_val': 600,
        'upper_val': 700,
        'step': 50,
        'min': 10,
        'max': 100000,
        'help': """Радиус цилиндрического резервуара, м""",
    },
    'pressure_initial': {
        'label': 'Начальное пластовое давление, атм',
        'lower_val': 100,
        'default_val': 500,
        'upper_val': 700,
        'step': 5,
        'min': 1,
        'max': 2000,
        'help': '',
    },
    'length_hor_well_bore': {
        'label': 'Длина горизонт. ствола, м',
        'lower_val': 100,
        'default_val': 200,
        'upper_val': 300,
        'step': 5,
        'min': 1,
        'max': 100000,
        'help': """Длина горизонтального ствола скважины, м""",
    },
    'length_half_fracture': {
        'label': 'Полудлина трещины ГРП xf, м',
        'lower_val': 10,
        'default_val': 60,
        'upper_val': 150,
        'step': 1,
        'min': 1,
        'max': 10000,
        'help': '',
    },
}

FTOR_DECODE = {
    'permeability': {
        'label': 'Проницаемость k, мД',
    },
    'skin': {
        'label': 'Skin',
    },
    'res_radius': {
        'label': 'Радиус цилиндрического резервуара, м',
    },
    'pressure_initial': {
        'label': 'Начальное пластовое давление, атм',
    },
    'length_hor_well_bore': {
        'label': 'Длина горизонт. ствола, м',
    },
    'length_half_fracture': {
        'label': 'Полудлина трещины ГРП xf, м',
    },
    "kind_code": {
        'label': 'Тип скважины',
        0: "Вертикальная скважина",
        1: "Горизонтальная скважина",
        2: "Вертикальная скважина с ГРП",
        3: "Горизонт. скв. МГРП",
    },
}
