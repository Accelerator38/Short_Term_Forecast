from .base import ClusteredBaseModel
from .crm import CRM
from ..optimization.minimizers import BaseMinimizer
from .parameters import inverse_distance_mtrx, ReverseSparseConverter, ParamsLinker, ExcelExporter

from collections import Counter

from scipy.sparse import csr_matrix, find
from numba import jit
import numpy as np
from logging import getLogger

logger = getLogger(__name__)


@jit('Tuple((f8[:, :], f8[:,:]))(f8[:, :], f8[:, :], f8[:, :], f8[:, :], f8[:, :], f8[:, :], f8[:, :], i8[:, :], i8)', nopython=True)
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

    n_injectors = f.shape[0]
    n_producers = f.shape[1]
    n_timesteps = inj.shape[0]

    single_volume_prod = np.zeros((n_injectors, n_producers), dtype=np.float64)
    ret_prod = np.zeros((n_timesteps, n_producers), dtype=np.float64)

    # iterate over each production well
    for p in range(n_producers):

        for i in range(n_injectors):
            # array for predicting production
            p_predict = np.zeros(n_timesteps, dtype=np.float64)
            prev_prod = prod_0[i, p]
            if is_halt[0, p]:
                p_predict[0] = .0
            else:
                p_predict[0] = prod_0[i, p]

            for t in range(1, n_timesteps):
                # precompute for efficiency
                if tau[i, p] == 0.0:
                    continue
                _tau_exp = np.exp(-1 / tau[i, p]) # np.reciprocal работает дольше

                if scheme == 0:  # SVIR | step injectivity, linear BHP
                    p_predict[t] = \
                        prev_prod * _tau_exp + \
                        (f[i, p] * inj[t, i] - J[i, p] * tau[i, p] * beta[t - 1, p]) * (1 - _tau_exp)

                if p_predict[t] < 0 or is_halt[t, p]:
                    p_predict[t] = 0.0
                    prev_prod = p_predict[t - 1] if p_predict[t - 1] != .0 else prev_prod
                else:
                    prev_prod = p_predict[t]


            single_volume_prod[i, p] = p_predict[-1]

            ret_prod[:, p] += p_predict

    # single_volume_prod is necessary for initial guess in testing
    return ret_prod, single_volume_prod


def _loss_func(x, prod, inj, beta, gamma, is_halt, rsc, pl, scheme):
    prod_0_data, tau_data, J_data, f_data = pl.split(x)
    prod_0, tau, J, f = rsc.convert(prod_0_data), rsc.convert(tau_data), rsc.convert(J_data), rsc.convert(f_data)

    prod_p, _ = _predict(tau, J, f, prod_0, inj, beta, gamma, is_halt, scheme)

    return np.mean((prod - prod_p) ** 2)


class CRMIP(CRM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.scheme == 1: # 'LVIR'
            logger.error("LVIR for CRMIP has not implemented yet.")
            raise NotImplemented('LVIR for CRMIP has not implemented yet.')

    def _initial_guess(self, production):
        logger.info("Evaluating initial guess")

        prod_cnt, inj_cnt = len(self.prod_n), len(self.inj_n)

        tau_0 = np.ones((inj_cnt, prod_cnt)) * self.tau_0
        J_0 = np.ones((inj_cnt, prod_cnt)) * self.J_0

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
        X, Y, f_data = find(f_0)
        self.f_size = len(f_data)
        self.f_bnds = [(0, 1) for _ in range(len(f_data))]
        logger.info(f"Sparse f_0 was formed, size={len(f_data)}")

        if self.iprod_0 is None:
           prod_0 = np.ones((inj_cnt, prod_cnt)) * production[0]
           for i, cnt in Counter(Y).items():
               prod_0[:, i] /= cnt
        else:
           prod_0 = np.ones((len(self.inj_n), len(self.prod_n))) * self.iprod_0

        self.rsc = ReverseSparseConverter((X, Y), f_0.get_shape())
        J_0_data, tau_0_data, prod_0_data = J_0[self.rsc.indices], tau_0[self.rsc.indices],prod_0[self.rsc.indices]
        self.J_size, self.tau_size, self.prod_size = len(J_0_data), len(tau_0_data), len(prod_0_data)
        logger.info(f"\ttau_0 (shape={tau_0.shape}): {self.rsc.convert(tau_0_data)}")
        logger.info(f"\tJ_0 (shape={J_0.shape}: {self.rsc.convert(J_0_data)}")

        self.pl = ParamsLinker(prod_0_data, tau_0_data, J_0_data, f_data)
        return self.pl.link(prod_0_data, tau_0_data, J_0_data, f_data)

    def get_loss_func(self, production, is_halt, injection, beta, gamma):
        return lambda x: _loss_func(x, production, injection, beta, gamma,
                                    is_halt, self.rsc, self.pl, scheme=self.scheme)

    def unpack(self, x):
        prod_0_data, tau_data, J_data, f_data = self.pl.split(x)
        prod_0, tau, J, f = self.rsc.convert(prod_0_data), self.rsc.convert(tau_data), \
                            self.rsc.convert(J_data), self.rsc.convert(f_data)
        return prod_0, tau, J, f

    def fit(self, production, is_halt, injection, pressures, minimizer: BaseMinimizer):
        ret = super().fit(production, is_halt, injection, pressures, minimizer)
        gamma, beta = self._derivatives(injection, pressures)
        _, self._last_single_volume_prod = _predict(self.tau, self.J, self.f, self.prod_0, injection, beta, gamma,
                                                    is_halt, self.scheme)
        return ret

    def predict(self, prod_0, is_halt, injection, pressures):
        gamma, beta = self._derivatives(injection, pressures)
        return _predict(self.tau, self.J, self.f, prod_0, injection, beta, gamma, is_halt, scheme=self.scheme)[0]

    def to_excel(self, file_path='CRMIP parameters.xlsx'):
        with ExcelExporter(file_path) as excel_exporter:
            excel_exporter.write_matrix_to_worksheet('tau', self.inj_n, self.tau, self.prod_n)
            excel_exporter.write_matrix_to_worksheet('J', self.inj_n, self.J, self.prod_n)
            excel_exporter.write_matrix_to_worksheet('f', self.inj_n, self.f, self.prod_n)


class ClusteredCRMIP(CRMIP, ClusteredBaseModel):

    def __init__(self, prod_to_cluster, inj_to_cluster, *args, **kwards):
        super().__init__(*args, **kwards)
        self.cross_params_mask = (inj_to_cluster[:, np.newaxis] == prod_to_cluster[np.newaxis, ]) * prod_to_cluster
        self.prod_to_cluster = prod_to_cluster
        self.inj_to_cluster = inj_to_cluster

    def get_x_mask(self, cluster_n):
        if self.pl is None or self.rsc is None:
            raise Exception('Used before x_0 initialization')

        cross_params_clust_mask = self.cross_params_mask == cluster_n
        cross_params_clust_flatten = cross_params_clust_mask[self.rsc.indices]
        return self.pl.link(*([cross_params_clust_flatten]*4))