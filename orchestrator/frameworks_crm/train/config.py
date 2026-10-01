import numpy as np
from os import path
import datetime
# info for prepare_feather_data
oilfield = 'Восточно-Мессояхское'
date_start = datetime.date(2018, 1, 1)
date_test = datetime.date(2021, 1, 31)
date_end = datetime.date(2021, 4, 30)

# %% Columns to read
prod_columns = [
    'well.ois',
    'dt',
    # 'Давление забойное от Hд',
    # 'Давление забойное от Pпр',
    'Давление забойное',
    'Дебит жидкости среднесуточный',
    'Дебит нефти расчетный',
    # 'Обводненность объемная',
    # 'Обводненность (ТМ)',
    # 'Обводненность (ХАЛ)',
    'charwork.name',
    'sost',
]

inj_columns = [
    'well.ois',
    'dt',
    # 'Приемистость ТМ',
    # 'Приемистость исходная по ТМ',
    # 'Приемистость сервисных служб',
    'Приемистость среднесуточная',
    'charwork.name',
    'sost',
]

preset_choice = 'messo'
presets = \
    {
        'otdelnoe': {'data': 'Отдельное', 'models': 'models_otdelnoe_test2',
                  'result': 'result_otdelnoe_test2',
                  'CUT_DATE': '2018-01-01', 'CUT_DATE2': '2021-01-30',
                  'batch_size': 128},

        'romanovskoe': {'data': 'Романовское', 'models': 'models_romanovskoe_days_200',
                  'result': 'result_romanovskoe_days_200',
                  'CUT_DATE': '2018-01-01', 'CUT_DATE2': '2020-11-30',
                  'batch_size': 128},

        'messo': {'data': 'Восточно-Мессояхское', 'models': 'models_messo_aq_double_r_500',
                  'result': 'result_messo_aq_double_r_500',
                  'CUT_DATE': '2018-01-01', 'CUT_DATE2': '2020-11-30',
                  'batch_size': 128},
     }

# Рабочая директория обучения
path_main_train = path.join(path.dirname(path.abspath(__file__)), '')
print(path_main_train)
# Расположение исходных файлов
path_train = path.join(path_main_train, presets[preset_choice]['data'])

# Сохранение моделей
models_save_path_train = path.join(path_main_train, presets[preset_choice]['models'])

# куда сохраняется "выход" пайплана
save_path_output_pipeline_train = path.join(path_main_train, presets[preset_choice]['result'])

# Куда сохранять обученные модели
save_path_train = path.join(save_path_output_pipeline_train, 'CRM_output')

TR_DAY = 21
# В разное время может работать разное количество скважин. С целью получения более точных результатов, можно выбрать
# промежуток, в котором количество рабочих скважин ведет себя стабильно.

CUT_DATE = presets[preset_choice]['CUT_DATE']
CUT_DATE2 = presets[preset_choice]['CUT_DATE2']

split_ratio = 0.8

test_period_len = ((1-split_ratio) * (datetime.datetime.strptime(CUT_DATE2, '%Y-%m-%d') -
                                datetime.datetime.strptime(CUT_DATE, '%Y-%m-%d'))).days - 1

print('num_test_days', test_period_len)

# Для моделей CRM/CRMIP
display_graphs = True
# схема модели svir/lwir
SCHEME ='LVIR' # 'SVIR'
# начальное значение параметров модели
TAU_0 = 50
J_0 = 1
# радиус влияния (вне его скважины считаются невзаимодействующими)
INFLUENCE_R = 750
# учёт радиуса от хвоста и пятки скважины
double_R = True
inactive_time = 0.8

# какие модели строить
liquid_flag = wc_flag = gas_factor_flag = pressure_reservoir_flag = True
# Параметры оптимизации SLSQP
# maxiter - максимальное количество итераций оптимизатора
# ftol - минимальное изменение целевой функции, при меньшем значении которого оптимизация останавливается
# eps - шаг для численной оценки якобиана
# disp -
options_SLSQP_CRM = {'maxiter': 50, 'ftol': 1e-7, 'eps': 1.4901161193847656e-8, 'disp': True}

# ### CRM, CRMIP
# Ограничения на оптимизируемые параметры
# границы для параметров для CRM/CRMIP
prod_bnd = (0, 500)
tau_bnd = (1e-6, 3000)
J_bnd = (1e-6, 10)
f_bnd = (0, 1)
bnds_CRM = {'prod_bnd': prod_bnd, 'tau_bnd': tau_bnd, 'J_bnd': J_bnd, 'f_bnd': f_bnd}

# TODO
# Функция востановления давления
# p_res - пластовое давление
# p_0 -
# recov_time - характерное время восстановления давления
p_res = 220
p_0 = 10
recov_time = 100

# Tpot config
generations = 30
population_size = 50
early_stop = 5

# generations = 1
# population_size = 1
# early_stop = 1

# for plotting
border = 0.001
# LSTM
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FILES_PATH = r'/Users/stas/Documents/NTC/LSTM/preporation/Otdelnoe/ToLSTM'
# Constants declaration
NUM_NEURONS_LSTM1 = 250
NUM_NEURONS_LSTM2 = 250
NUM_NEURONS_DENSE = 200
# EPOCHS = 5
# BATCH_SIZE = 8

EPOCHS = 5
BATCH_SIZE = presets[preset_choice]['batch_size']
# BATCH_SIZE = 2048

# на основе скольких дней делается предсказание
time_steps = 10
# scaler = MinMaxScaler(copy=True, feature_range=(0.1, 1))
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
