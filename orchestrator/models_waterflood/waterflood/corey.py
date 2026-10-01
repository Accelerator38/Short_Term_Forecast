import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, dual_annealing, minimize_scalar, shgo
from typing import Dict


class CoreyModel(object):
    """Модель обводненности скважины по Кори.

    Модель основана на уравнении Дюпюи, модифицированных ОФП Брукса-Кори и
    предположении, что скважина дренирует определенный объем пластового
    резервуара, где наблюдается двухфазная фильтрация вода-нефть. Модель
    характеризует долю воды в притоке к забою скважины в зависимости от КИН
    этого объема.

    Перед использованием модель необходимо построить. Построение модели может
    быть выполнено путем прямого задания значений параметров или с помощью
    процедуры адаптации на фактические значения обводненности. Адаптация модели
    включает 2 последовательных шага: адаптация на общий тренд, адаптация на
    последние значения.

    Parameters
    ----------
    optimizer_trend_name: {"diff", "dual", "shgo"}, default='diff'
        Оптимизационный алгоритм для использования на первом шаге адптации.

    n_last_minimize: int, default=10
        Количество последних фактических точек обводненности, на которых
        выполняется последний шаг адаптации модели.

    Notes
    -----
    Параметры модели

    **watercut_initial**: float, диапазон изменения: [-0.5, 1]
        Фиктивное начальное значение обводненности, д. ед. Данный параметр не
        имеет физического смысла, необходим только для лучшей регуляции модели
        относительно фактических значений обводненности.

    **mobility_ratio**: float, диапазон изменения: [0.0025, 50]
        Соотношение подвижностей нефти и воды в пластовых условиях, безразм.

        .. math:: (k^max_ro / \mu_o) / (k^max_rw /  \mu_w)

    **n_o**: float, диапазон изменения: [1, 6]
        Степень по нефти в корреляции ОФП Брукса-Кори, безразм.

    **n_w**: float, диапазон изменения: [1, 6]
        Степень по воде в корреляции ОФП Брукса-Кори, безразм.

    **stooip**: float
        Геологические запасы нефти дренируемого скважиной объема в поверхностных
        условиях, тыс. м3.

    References
    ----------

    .. [1] https://onepetro.org/SPEAPOG/proceedings-abstract/06APOGCE/All-06APOGCE/SPE-101144-MS/141528

    .. [2] https://onepetro.org/SPERPTC/proceedings-abstract/08ROGC/All-08ROGC/SPE-115497-MS/144701

    .. [3] https://petrowiki.spe.org/Relative_permeability_models

    """
    _param_names = [
        'watercut_initial',
        'mobility_ratio',
        'n_o',
        'n_w',
        'stooip',
    ]
    _optimizers = {
        'diff': differential_evolution,
        'dual': dual_annealing,
        'shgo': shgo,
    }

    def __init__(
            self,
            optimizer_trend_name: str = 'diff',
            n_last_minimize: int = 10,
    ):
        self._optimizer = self._optimizers[optimizer_trend_name]
        self._n_last_minimize = n_last_minimize
        self._params = dict.fromkeys(self._param_names)
        self._cums_oil = None
        self._watercuts = None

    def get_params(self) -> Dict[str, float]:
        """Возвращает значения параметров модели."""
        return self._params

    def set_params(self, *params_list, **params_dict) -> None:
        """Устанавливает значения параметров модели."""
        self._params.update(dict(zip(self._param_names, params_list)))
        self._params.update(params_dict)

    def compute(self, cums_oil: np.ndarray) -> np.ndarray:
        """Считает обводненность в зависимотси от накопленной добычи нефти.

        Рассчитываются значения обводненности скважины по соответсвующим
        моментам времени накопленной добычи нефти.

        Parameters
        ----------
        cums_oil
            Значения накопленной добычи нефти, тыс. м3.

        Returns
        -------
            Значения обводненности, д. ед.
        """
        recovery_factor = cums_oil / (self._params['stooip'] * 1e6)
        term_1 = (1 - recovery_factor) ** self._params['n_o']
        term_2 = self._params['mobility_ratio'] * recovery_factor ** self._params['n_w']
        watercut = self._params['watercut_initial'] + 1 / (1 + term_1 / term_2)
        return watercut

    def fit(self, cums_oil: np.ndarray, watercuts: np.ndarray) -> None:
        """Выполняет адаптацию модели на фактические значения.

        Parameters
        ----------
        cums_oil
            Значения накопленной добычи нефти, тыс. м3.
        watercuts
            Значения обводненности, д. ед.
        """
        self._cums_oil = cums_oil
        self._watercuts = watercuts
        self._set_param_bounds()
        bounds = list(tuple(min_max.values()) for min_max in self._param_bounds.values())
        result_full = self._optimizer(
            self._loss_function_trend_full,
            bounds=bounds,
        )
        result_last = minimize_scalar(
            self._loss_function_trend_last,
            bounds=bounds[0],
            method='Bounded',
        )
        self.set_params(*result_full.x)
        self.set_params(watercut_initial=result_last.x)

    def predict(self, cums_liq: np.ndarray, cum_oil_start: float) -> np.ndarray:
        """Выполняет прогнозирование обводненности.

        Рассчитываются значения обводненности скважины по соответствующим
        моментам времени накопленной добычи жидкости.

        Parameters
        ----------
        cums_liq
            Значения накопленной добычи жидкости, тыс. м3.
        cum_oil_start
            Значение накопленной добычи нефти перед прогнозированием, тыс. м3.
        Returns
        -------
            Значения обводненности, д. ед.
        """
        solution = solve_ivp(
            fun=lambda cum_liq, cum_oil: 1 - self.compute(cum_oil),
            vectorized=True,
            y0=[cum_oil_start],
            t_span=(cums_liq[0], cums_liq[-1]),
            t_eval=cums_liq,
            method='RK45',
        )
        cums_oil = solution.y[0]
        watercuts = self.compute(cums_oil)
        return watercuts

    def _set_param_bounds(self) -> None:
        cum_max = max(self._cums_oil) / 1e3
        self._param_bounds = {
            'watercut_initial': {'min': -0.5, 'max': 1},
            'mobility_ratio': {'min': 0.0025, 'max': 50},
            'n_o': {'min': 1, 'max': 6},
            'n_w': {'min': 1, 'max': 6},
            'stooip': {'min': cum_max / 0.8, 'max': cum_max / 0.05},
        }

    def _loss_function_trend_full(self, params: np.ndarray) -> float:
        self.set_params(*params)
        err = self._calc_error(
            self._cums_oil,
            self._watercuts,
        )
        return err

    def _loss_function_trend_last(self, watercut_initial: float) -> float:
        self.set_params(watercut_initial=watercut_initial)
        err = self._calc_error(
            self._cums_oil[-self._n_last_minimize:],
            self._watercuts[-self._n_last_minimize:],
        )
        return err

    def _calc_error(self, cums_oil: np.ndarray, watercuts_true: np.ndarray) -> float:
        watercuts_pred = self.compute(cums_oil)
        mae = float(np.mean(abs(watercuts_true - watercuts_pred)))
        return mae
