import datetime


class Config(object):
    """Настройки расчета для конкретного месторождения.

    Parameters
    ----------
    field_name: str
        Название месторождения на русском.

    shops: list[str]
        Список названий цехов на русском.

    date_start: datetime.date
        Дата начала адаптации моделей (начало с 00:00 этой даты).

    date_test: datetime.date
        Дата начала прогноза моделей (начало с 00:00 этой даты).

    date_end: datetime.date
        Дата конца прогноза моделей (конец по 23:59 этой даты).

    """
    def __init__(
            self,
            field_name: str,
            shops: list[str],
            date_start: datetime.date,
            date_test: datetime.date,
            date_end: datetime.date,
    ):
        self.field_name = field_name
        self.shops = shops
        self.date_start = date_start
        self.date_test = date_test
        self.date_end = date_end
