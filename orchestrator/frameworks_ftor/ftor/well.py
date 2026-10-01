import datetime
import pandas as pd


class WellResults(object):

    def __init__(
            self,
            all_params: list[dict[str, float]],
            adap_and_fixed_params: list[dict[str, float]],
            params_model_corey: dict[str, float],
            rates_liq_train: pd.Series,
            rates_liq_test: pd.Series,
            watercuts_train: pd.Series,
            watercuts_test: pd.Series,
            rates_oil_test: pd.Series,
    ):
        self.all_params = all_params
        self.adap_and_fixed_params = adap_and_fixed_params
        self.params_model_corey = params_model_corey
        self.rates_liq_train = rates_liq_train
        self.rates_liq_test = rates_liq_test
        self.watercuts_train = watercuts_train
        self.watercuts_test = watercuts_test
        self.rates_oil_test = rates_oil_test


class Well(object):
    """Скважина для расчета.
    """
    NAME_STATUS = 'Состояние'
    NAME_EVENT = 'Мероприятие'
    NAME_CONSTRAINTS = 'Ограничения параметров'
    NAME_PRESSURE = 'Давление забойное'
    NAME_WATERCUT = 'Обводненность объемная'
    NAME_RATE_LIQ = 'Дебит жидкости'
    NAME_RATE_OIL = 'Дебит нефти'
    NAME_CUM_LIQ = 'Добыча жидкости накопленная'
    NAME_CUM_OIL = 'Добыча нефти накопленная'

    def __init__(
            self,
            well_name: int,
            date_start: datetime.date,
            date_test: datetime.date,
            date_end: datetime.date,
            thickness: float,
            porosity: float,
            compressibility_total: float,
            density_oil: float,
            viscosity_liq: float,
            volume_factor_liq: float,
            cum_liq_start: float,
            cum_liq_test: float,
            cum_oil_test: float,
            df_chess: pd.DataFrame,
            df_flood: pd.DataFrame,
    ):
        self.well_name = well_name
        self.date_start = date_start
        self.date_test = date_test
        self.date_end = date_end
        self.thickness = thickness
        self.porosity = porosity
        self.compressibility_total = compressibility_total
        self.density_oil = density_oil
        self.viscosity_liq = viscosity_liq
        self.volume_factor_liq = volume_factor_liq
        self.cum_liq_start = cum_liq_start
        self.cum_liq_test = cum_liq_test
        self.cum_oil_test = cum_oil_test
        self.df_chess = df_chess
        self.df_flood = df_flood
        self._results = None

    @property
    def results(self) -> WellResults:
        return self._results

    @results.setter
    def results(self, results: WellResults) -> None:
        self._results = results
