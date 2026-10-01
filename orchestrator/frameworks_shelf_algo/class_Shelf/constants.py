DEBIT = 'Дебит нефти расчетный'
DEBIT1 = 'Дебит жидкости среднесуточный'
DEBIT_GAS = 'Дебит газа (ТМ)'
STATUS = 'sost'
DATE_FORMAT = '%Y-%m-%d'
DEC_RATES = 'DEC_RATES'
DEC_RATES_LIQ = 'DEC_RATES_LIQ'
DEC_RATES_GAS = 'DEC_RATES_GAS'
LAST_MEASUREMENT = 'LAST_MEASUREMENT'
DATA = 'данные'
DATE = 'DATE'
VALUE = 'VALUE'
VALUE_LIQ = 'VALUE_LIQ'
VALUE_GAS = 'VALUE_GAS'
GTMS = 'GTMS'
NAME = 'NAME'
MLSP_STOP = 'MLSP_STOP'
DATE_START_MLSP = "Дата запуска МЛСП"
DEBIT_INCREASE = "Прирост к доостановочному дебиту нефти"
DEBIT_INCREASE_LIQ = "Прирост к доостановочному дебиту жидкости"
DEBIT_INCREASE_GAS = "Прирост к доостановочному дебиту газа"
N_DAYS_DEBIT_RECOVERY = "Продолжительность вывода на доостановочный дебит"
TARGET_DEBIT_AFTER_MLSP_START = 'Целевой дебит нефти ВНР после останова МЛСП'
TARGET_DEBIT_AFTER_MLSP_START_LIQ = 'Целевой дебит жидкости ВНР после останова МЛСП'
DEC_RATE_DURING_MLSP_VNR = 'Темп падения после останова МЛСП'
DEC_RATE_DURING_MLSP_VNR_LIQ = 'Темп падения после останова МЛСП'
MLSP_VNR_FIRST_DATE = 'Первый день периода выхода на режим после останова МЛСП'
MLSP_VNR_LAST_DATE = 'Последний день периода выхода на режим после останова МЛСП'
MLSP_VNR_IS_GOING = 'Идет вывод на режим после останова МЛСП'
NEW_WELL = 'Новая скважина'
PLANNED_MLSP_STOPS = "MLSP_STOP"
RESULTS = 'Результаты'
GTM_DATA_FORMAT = {
    # 'Ввод новых скважин': {
    #     'название': 'str',
    #     'OIS': 'int',
    #     'дебит': 'float',
    #     'длительность выхода на режим': 'int'
    # },
    # 'Current workover of wells': {
    #     'Flow rate during CWW': 'float',
    #     'Duration of CWW': 'int',
    #     'Duration of recovering': 'int'
    # },
    'Текущий ремонт скважин': {
        'дебит нефти в период ТРС': 'float',
        'дебит жидкости в период ТРС': 'float',
        'длительность ТРС': 'int',
        'длительность выхода на режим': 'int'
    },
    'Капитальный ремонт скважин': {
        'дебит нефти в период КРС': 'float',
        'дебит жидкости в период КРС': 'float',
        'эффект по нефти от КРС': 'float',
        'эффект по жидкости от КРС': 'float',
        'длительность КРС': 'int',
        'длительность выхода на режим': 'int'
    },
    'Перевод в нагнетательный фонд': {
        'дебит нефти в период и после работ': 'float',
        'дебит жидкости в период и после работ': 'float'
    },
    'Соляно-кислотная обработка': {
        'дебит нефти в период СКО': 'float',
        'дебит жидкости в период СКО': 'float',
        'длительность СКО': 'int',
        'эффект по нефти от СКО': 'float',
        'эффект по жидкости от СКО': 'float',
    },
    'Промыслово-геофизические исследования': {
        'дебит нефти во время остановки': 'float',
        'дебит жидкости во время остановки': 'float',
        'длительность остановки': 'int',
    }
}
