from ..optimization.minimizers import BaseMinimizer
from .parameters import inverse_distance_mtrx, ReverseSparseConverter, ParamsLinker, ExcelExporter
from .base import BaseModel, ClusteredBaseModel

from scipy.sparse import csr_matrix, find
from numba import jit
import numpy as np
from itertools import product
from logging import getLogger

logger = getLogger(__name__)


@jit('f8[:,:](f8[:], f8[:], f8[:, :], f8[:], f8[:, :], f8[:, :], f8[:, :], i8[:, :], i8)', nopython=True) #для оптимизации работы кода (numba)
def _predict(tau, J, f, prod_0, inj, beta, gamma, is_halt, scheme):
    # tau - постоянная времени, параметр модели, непосредственносвязанный с фильтрационно-емкостными свойствами пласта и определяющий характерное время фильтрации пластовых флюидов.
    # J - коэффициент продуктивности добывающей скважины
    # f - доля дебита 𝑗-й нагнетающей скважины, приходящаяся на 𝑖-ю добывающую
    # prod_0 - дебит i-ой скважины в начальный момент времени
    # inj - приемистость 𝑘-й нагнетательной скважины в момент времени
    # beta - производная забойного давления на𝑘-м шаге по времени
    # gamma - производная приемистости на𝑘-м шаге по времени
    # is_halt - работа\неработа скважины
    # scheme - схема LVIR\SVIR
    n_producers = f.shape[1]
    n_timesteps = inj.shape[0]

    ret_prod = np.zeros((n_timesteps, n_producers), dtype=np.float64)

    # iterate over each production well (перебор каждой добывающий скважины)
    for p in range(n_producers):
        # array for predicting production (массив для прогнозирования добычи)
        p_predict = np.zeros(n_timesteps, dtype=np.float64)

        prev_prod = prod_0[p] # оценка дебита i-ой скважины в начальный момент времени
        if is_halt[0, p]: # учёт остановки
            p_predict[0] = .0 #если была остановка - заполняется нулями
        else:
            p_predict[0] = prod_0[p] # Если остановки не было - заполняем значениями

        for t in range(1, n_timesteps):
            # precompute for efficiency (Предварительный расчёт для повышения эффективности)
            _tau_exp = np.exp(-1 / tau[p]) # np.reciprocal работает дольше

            # crmp formula Формула написана правильно (надо опираться на дипломную работу Цанда Артёма, в других статьях формулы другие )

            if scheme == 1: # LVIR | linear injectivity, linear BHP | LVIR - линейное изменение закачки в пределах заданного интервала времени.
                p_predict[t] = \
                    prev_prod * _tau_exp + \
                    (np.sum(inj[t] * f[:, p]) - np.sum(inj[t - 1] * f[:, p]) * _tau_exp) - \
                    (np.sum(gamma[t - 1] * f[:, p]) + J[p] * beta[t - 1, p]) * tau[p] * (1 - _tau_exp)

            elif scheme == 0: # SVIR | step injectivity, linear BHP | SVIR - последовательное ступенчатое изменение
                                                                            # закачки при линейной вариации забойного давления
                p_predict[t] = \
                    prev_prod * _tau_exp + \
                    (np.sum(inj[t] * f[:, p]) - J[p] * tau[p] * beta[t - 1, p]) * (1 - _tau_exp)

            if p_predict[t] < 0 or is_halt[t, p]: # проверка заполнения значений с учётом остановки
                p_predict[t] = 0.0
                prev_prod = p_predict[t - 1] if p_predict[t - 1] != .0 else prev_prod
            else:
                prev_prod = p_predict[t]


        ret_prod[:, p] = p_predict

    return ret_prod


def _loss_func(x, prod, inj, beta, gamma, is_halt, rsc, pl, scheme):
    prod_0, tau, J, f_data = pl.split(x)
    f = rsc.convert(f_data)

    prod_p = _predict(tau, J, f, prod_0, inj, beta, gamma, is_halt, scheme)

    return np.mean((prod - prod_p) ** 2)


class CRM(BaseModel):

    def __init__(self, coordinates, bnds, prod_n, inj_n, scheme='LVIR', tau_0=50, J_0=10, iprod_0=None, influence_r=2*1e3, double_r=False):
        self.bnds = bnds
        self.coordinates = coordinates

        self.prod_n = prod_n
        self.inj_n = inj_n
        self.tau_0 = tau_0
        self.J_0 = J_0
        self.iprod_0 = iprod_0
        self.influence_r = influence_r
        self.double_r = double_r

        if scheme not in ['LVIR', 'SVIR']: # кодировка схем для numba
            logger.error("Invalid value in scheme, LVIR or SVIR are only acceptable.")
            raise ValueError('Invalid value in scheme, LVIR or SVIR are only acceptable.')
        # numba does not support strings, so we need to encode values
        # Let it be following
        # 'SVIR': 0
        # 'LVIR': 1
        self.scheme = int(scheme == 'LVIR')

    def f_sum_constraint(self, x): # ограничение на f_sum
        if self.pl is None or self.rsc is None:
            raise Exception('Used before x_0 initialization')
        f_data = self.pl.split(x)[-1]
        f = self.rsc.convert(f_data)
        return np.min(1 - np.sum(f, axis=1))

    # рассчёт расстояния для двойного радиуса
    def dist(x1, y1, x2, y2):
        return np.sqrt((x1-x2) ** 2 + (y1 - y2) ** 2)

    def _check_for_nans(self, *args):# проверка на nan
        for arg in args:
            assert not np.any(np.isnan(arg)), f'Nans in {arg}'

    def _initial_guess(self, production): # оценка первоначального предположения
        logger.info("Evaluating initial guess")
        # May be it's not the best solution - почему не лучшее решение???
        if self.iprod_0 is None: #проверяем iprod_0
            prod_0 = production[0]
        else:
            prod_0 = np.ones(len(self.prod_n)) * self.iprod_0 # заполняем массив единицами и умножаем его на параметр iprod_0
        self.prod_size = prod_0.size

        tau_0 = np.ones(len(self.prod_n)) * self.tau_0 # проделываем аналогичные действия для проверки тау, джей и ф нулевое
        self.tau_size = tau_0.size
        logger.info(f"\ttau_0 (shape={tau_0.shape}): {tau_0}")
        J_0 = np.ones(len(self.prod_n)) * self.J_0
        self.J_size = J_0.size
        logger.info(f"\tJ_0 (shape={J_0.shape}: {J_0}")
        ### учёт двойного радиуса
        if self.double_r==False:
            f_0 = inverse_distance_mtrx(self.coordinates[['Координата X', 'Координата Y']].loc[self.prod_n],
                                        self.coordinates[['Координата X', 'Координата Y']].loc[self.inj_n],
                                        dist_thres=self.influence_r)
        else:
            f_0 = np.full((len(self.inj_n), len(self.prod_n)), self.influence_r+1)
            for inj in range(f_0.shape[0]):
                for prod in range(f_0.shape[1]):
                    prod_x1, prod_x2 = self.coordinates.loc[self.prod_n[prod]][['Координата X', 'Координата X_1']]
                    prod_y1, prod_y2 = self.coordinates.loc[self.prod_n[prod]][['Координата Y', 'Координата Y_1']]
                    inj_x1, inj_x2 = self.coordinates.loc[self.inj_n[inj]][['Координата X', 'Координата X_1']]
                    inj_y1, inj_y2 = self.coordinates.loc[self.inj_n[inj]][['Координата Y', 'Координата Y_1']]

                    # угловой коэффициент для добывающего и нагнетательного столба
                    try:
                        k_prod = (prod_y2 - prod_y1) / (prod_x2 - prod_x1)
                    except ZeroDivisionError:
                        k_prod = 0
                    try:
                        k_inj = (inj_y2 - inj_y1) / (inj_x2 - inj_x1)
                    except ZeroDivisionError:
                        k_inj = 0

                    # перебираем координаты по иксу и находим расстояние между двумя точками
                    for i in np.linspace(prod_x1, prod_x2, 10):
                        for j in np.linspace(inj_x1, inj_x2, 10):
                            res = np.sqrt((i-j)**2 +
                                          ((k_prod * (i - prod_x1) + prod_y1) - (k_inj * (j - inj_x1) + inj_y1))**2)
                            if res < f_0[inj][prod]:
                                f_0[inj][prod] = res
            f_0[f_0 >= self.influence_r] = 0

            with np.errstate(divide='ignore', invalid='ignore'):
                f_0 = 1. / f_0
                f_0[f_0 == np.inf] = 0.0
        #####
        logger.info(f"\t full f_0 evaluated, shape={f_0.shape}, size={f_0.size}")
        f_0 = csr_matrix(f_0)
        X, Y, f_data = find(f_0) # координаты
        self.f_size = len(f_data)
        logger.info(f"Sparse f_0 was formed, size={len(f_data)}")

        self.rsc = ReverseSparseConverter((X, Y), f_0.get_shape())
        self.pl = ParamsLinker(prod_0, tau_0, J_0, f_data)
        return self.pl.link(prod_0, tau_0, J_0, f_data)

    def _derivatives(self, injection, pressures): # рассчёт производных
        gamma = injection[1:, :] - injection[:-1, :]
        beta = pressures[1:, :] - pressures[:-1, :]
        return gamma, beta

    def get_loss_func(self, production, is_halt, injection, beta, gamma): # функция потерь - необходима для изменения функции потерь в наследованных классах
        """
        Function is necessary for changing loss function in inherited classes.
        Parameters
        ----------
        production
        is_halt
        injection
        beta
        gamma

        Returns
        -------

        """
        return lambda x: _loss_func(x, production, injection, beta, gamma,
                                         is_halt, self.rsc, self.pl, scheme=self.scheme) # используется лямбда-функция

    def unpack(self, x):
        # анпакер данных
        prod_0, tau, J, f_data = self.pl.split(x)
        f = self.rsc.convert(f_data)
        return prod_0, tau, J, f

    def fit(self, production, is_halt, injection, pressures, minimizer: BaseMinimizer): # функция обучения
        # инициализация параметров
        self._check_for_nans(production, is_halt, injection, pressures) # проверка на бесконечсноть добывающих, на остановку, нагнетающих и давления
        gamma, beta = self._derivatives(injection, pressures) # учёт производных гамма и бетта от нагнетательных и давления
        x_0 = self._initial_guess(production) # проверка первоначалььной гипотезы от добывающих значений
        # оптимизация с помощью функции ошибки и указания границ
        loss_func = self.get_loss_func(production, is_halt, injection, beta, gamma)
        bounds = [self.bnds['prod_bnd']] * self.prod_size + \
                 [self.bnds['tau_bnd']] * self.tau_size + \
                 [self.bnds['J_bnd']] * self.J_size + \
                 [self.bnds['f_bnd']] * self.f_size
        assert len(bounds) == x_0.size
        minimizer.set_bounds(bounds) # минимайзер для функции потерь, для границ
        minimizer.set_constraints([{'type': 'ineq', 'fun': self.f_sum_constraint}]) # минимайзер для функции потерь, для ограничений

        self.x_opt = minimizer.minimize(loss_func, False, x_0)
        # сохранение результатов
        self.prod_0, self.tau, self.J, self.f = self.unpack(self.x_opt)

        return self

    def predict(self, prod_0, is_halt, injection, pressures): # предикт параметров
        gamma, beta = self._derivatives(injection, pressures)
        return _predict(self.tau, self.J, self.f, prod_0, injection, beta, gamma, is_halt, scheme=self.scheme)

    def to_excel(self, file_path='CRM_parameters.xlsx'):
        with ExcelExporter(file_path) as excel_exporter:
            excel_exporter.write_series_to_worksheet('tau', self.prod_n, self.tau, ['Well', 'tau'])
            excel_exporter.write_series_to_worksheet('J', self.prod_n, self.J, ['Well', 'J'])
            excel_exporter.write_matrix_to_worksheet('f', self.inj_n, self.f, self.prod_n)


class ClusteredCRM(CRM, ClusteredBaseModel): # Кластеризация СРМ, нужна как регуляризация, чтобы оптимизировать процесс расчёта при понижении радиуса взаимовлияния.

    def __init__(self, prod_to_cluster, inj_to_cluster, *args, **kwards):
        super().__init__(*args, **kwards)
        self.f_mask = (inj_to_cluster[:, np.newaxis] == prod_to_cluster[np.newaxis, ]) * prod_to_cluster
        self.prod_to_cluster = prod_to_cluster
        self.inj_to_cluster = inj_to_cluster

    def get_x_mask(self, cluster_n):
        if self.pl is None or self.rsc is None:
            raise Exception('Used before x_0 initialization')

        prod_clust_mask = np.array(self.prod_to_cluster == cluster_n, dtype=int)
        f_clust_mask = np.array(self.f_mask == cluster_n, dtype=int)
        f_clust_flatten = f_clust_mask[self.rsc.indices]
        return self.pl.link(prod_clust_mask, prod_clust_mask, prod_clust_mask, f_clust_flatten).astype(bool)