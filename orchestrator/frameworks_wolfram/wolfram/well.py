import pandas as pd


class WellResults(object):

    def __init__(
            self,
            rates_liq_train: pd.Series,
            rates_liq_test: pd.Series,
            rates_oil_train: pd.Series,
            rates_oil_test: pd.Series,
            rates_gas_train: pd.Series,
            rates_gas_test: pd.Series
    ):
        self.rates_liq_train = rates_liq_train
        self.rates_liq_test = rates_liq_test
        self.rates_oil_train = rates_oil_train
        self.rates_oil_test = rates_oil_test
        self.rates_gas_train = rates_gas_train
        self.rates_gas_test = rates_gas_test


class Well(object):
    """Скважина для расчета.
    """
    NAME_PRESSURE = 'Давление забойное'
    NAME_RATE_LIQ = 'Дебит жидкости'
    NAME_RATE_OIL = 'Дебит нефти'
    NAME_RATE_BASE = 'Дебит базовый'
    NAME_RATE_GAS = 'Дебит газа'

    def __init__(
            self,
            well_name: int,
            df: pd.DataFrame,
    ):
        """Инициализация класса Well.

        Args:
            well_name: Номер скважины.
                Можно задать фиктивный номер, однако желательно указать реальный OIS номер.
            df: Данные скважины.
                Таблица должна содержать 4 последовательных столбца:
                    Давление забойное,
                    Дебит жидкости,
                    Дебит нефти,
                    Дебит газа.
                Значения указанных параметров в таблице могут быть в любых единицах измерения.
                Результат расчета будет представлен в тех же единицах измерения.

        """
        self.well_name = well_name
        self.df = df
        self._results = None

    @property
    def results(self) -> WellResults:
        return self._results

    @results.setter
    def results(self, results: WellResults) -> None:
        self._results = results
