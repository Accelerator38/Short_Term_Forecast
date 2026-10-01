from pandas import date_range


class ConfigStatistics:
    MODEL_NAMES = {
        "ftor": "Пьезо",
        "wolfram": "ML",
        "ensemble": "Ансамбль",
        "CRM": "CRM",
        "CRMIP": "CRMIP",
        "fedot": "CRM+ML",
        "true": "Факт",
        "Hybrid": "Hybrid",
        "Syntetic": "Syntetic",
        "shelf": "ППТП",
    }

    ignore_plots = [
        'Ошибка прогноза (нефть) "CRM"',
        'Ошибка прогноза (нефть) "CRMIP"',
        'Распределение ошибки (нефть) "CRM"',
        'Распределение ошибки (нефть) "CRMIP"',
        "Статистика по моделям",
    ]

    def __init__(
            self,
            oilfield: str,
            dates: date_range,
            use_abs: bool,
            well_names: tuple = (),
            bin_size: int = 10,
    ):
        self.oilfield = oilfield
        self.dates = dates
        self.use_abs = use_abs
        self.bin_size = bin_size
        self.well_names = well_names

    def exclude_wells(self, exclude_wells: list):
        self.well_names = [
            elem for elem in self.well_names if elem not in exclude_wells
        ]
