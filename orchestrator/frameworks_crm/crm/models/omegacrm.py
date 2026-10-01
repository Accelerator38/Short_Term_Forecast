import xlsxwriter
from numba import jit, prange
from scipy.sparse import csr_matrix, find
from ..optimization.minimizers import BaseMinimizer
from .parameters import inverse_distance_mtrx, ReverseSparseConverter, ParamsLinker, ExcelExporter
from .base import BaseModel
import numpy as np
from logging import getLogger

logger = getLogger(__name__)
# !TODO check model, reversion was done

@jit('i8(i8[:, :], i8, i8)', nopython=True) # отсылка к numba
def count_working_days(is_halt, well, n):# расчитывает рабочие дни
    count = 0
    i = n
    while not is_halt[i, well]:
        count += 1
        i -= 1
        if i < 0:
            count = -1
            break
    return count


@jit('Tuple((f8, f8[:], f8[:]))(i8, f8[:, :], f8[:, :, :], f8[:, :], i8)', nopython=True) #numba

def Q_n(wi, alpha, exp_omega, debit, n):
    wells_n = debit.shape[1]

    from_all_wells = .0
    dQ_dalpha, dQ_domega = np.zeros(wells_n), np.zeros(wells_n)
    for k in range(wells_n):
        if wi != k:
            from_one_well = .0  # np.sum(debit[0:n+1, k] * (exp_omega[wi, k, n+1:0:-1] - exp_omega[wi, k, n::-1]))
            for l in range(n + 1):
                w1, w2 = exp_omega[wi, k, n - l], exp_omega[wi, k, n - l + 1]
                dw1_domega = exp_omega[wi, k, n - l] / (n - l) if n != l else 0.0
                dw2_domega = exp_omega[wi, k, n - l + 1] / (n - l + 1)
                from_one_well += debit[l, k] * (w2 - w1)
                dQ_domega[k] += debit[l, k] * (dw1_domega - dw2_domega)

            dQ_dalpha[k] = from_one_well
            dQ_domega[k] *= alpha[wi, k]
            from_all_wells += alpha[wi, k] * from_one_well

    # поток жидкости должен быть отрицательным в случае добывающей скважины и положительным в случае закачки
    return -from_all_wells, -dQ_dalpha, -dQ_domega


@jit('Tuple((f8[:], f8[:], f8[:], f8[:, :], f8[:, :], f8[:, :]))(i8, f8[:], f8[:, :], f8[:, :, :], f8[:], f8, f8[:, :], f8[:, :], f8[:], i8[:, :], f8[:])', nopython=True)
def _predict_single_well(wi, tau, alpha, exp_omega, b, recovery_time, bhp, debit, p_initial, is_halt, J):
    size = bhp.shape[0]
    n_wells = exp_omega.shape[0]

    # Пластовое давление
    cont_pres = np.zeros(size)
    cont_pres[0] = p_initial[wi]

    # Предсказанный дебит
    pred_deb = np.zeros(size)
    pred_deb[0] = debit[0, wi]

    # Производные
    dtau, dalpha, domega, db = np.zeros(size, dtype=np.float64), np.zeros((size, n_wells), dtype=np.float64), \
                               np.zeros((size, n_wells), dtype=np.float64), np.zeros((size, 1), dtype=np.float64)
    dPdtau, dPdalpha, dPdomega = .0, np.zeros(n_wells, dtype=np.float64), np.zeros(n_wells, dtype=np.float64)
    dJ_db = .0

    exp_tau = np.exp(-1. / tau[wi])
    for n in range(1, size):
        work_time = count_working_days(is_halt, wi, n)
        # если не работает с самого начала то ничего не делаем, скважина еще не включилась
        if np.all(is_halt[:n + 1, wi]):
            cont_pres[n] = cont_pres[n - 1]
            pred_deb[n] = 0.
            continue

        # Расчет вклада соседей
        Q_n_1, dQ_dalpha, dQ_domega = Q_n(wi, alpha, exp_omega, debit, n - 1)

        # Чекалюк (нужен для поправки на кэоф. продуктивности)
        if is_halt[n, wi] or work_time == -1 or work_time > recovery_time:
            Jwi = J[wi]
        else:
            Jwi = J[wi] / (1 + b[0] * np.log(work_time / recovery_time))
            dJ_db = - J[wi] * np.log(work_time / recovery_time) / (1 + b[0] * np.log(work_time / recovery_time)) ** 2

        # Обновление пластового давления и дебита
        if is_halt[n, wi]:
            cont_pres[n] = cont_pres[n - 1] + Q_n_1 / (J[wi] * tau[wi])
            pred_deb[n] = 0.

            dPdtau = dPdtau - Q_n_1 / (J[wi] * tau[wi] ** 2)
            dPdalpha = dPdalpha + dQ_dalpha / (J[wi] * tau[wi])
            dPdomega = dPdomega + dQ_domega / (J[wi] * tau[wi])
        else:
            cont_pres[n] = exp_tau * cont_pres[n - 1] + (bhp[n, wi] + Q_n_1 / J[wi]) * (1 - exp_tau)
            pred_deb[n] = Jwi * (cont_pres[n] - bhp[n, wi])

            dPdtau = dPdtau * exp_tau + exp_tau * (cont_pres[n - 1] - bhp[n, wi] - Q_n_1 / J[wi]) / tau[wi] ** 2
            dtau[n] = Jwi * dPdtau
            dPdalpha = dPdalpha * exp_tau + (1 - exp_tau) * dQ_dalpha / J[wi]
            dalpha[n] = Jwi * dPdalpha
            dPdomega = dPdomega * exp_tau + (1 - exp_tau) * dQ_domega / J[wi]
            domega[n] = Jwi * dPdomega
            db[n] = dJ_db * (cont_pres[n] - bhp[n, wi])

    return pred_deb, cont_pres, dtau, dalpha, domega, db


@jit('Tuple((f8[:, :], f8[:, :], f8[:, :], f8[:, :, :], f8[:, :, :], f8[:, :, :]))(f8[:], f8[:, :], f8[:, :, :], f8[:], f8, f8[:, :], f8[:, :], f8[:], i8[:, :], f8[:])', nopython=True, parallel=True)
def _predict(tau, alpha, exp_omega, b, recovery_time, bhp, debit, p_initial, is_halt, J):
    n_wells = tau.size
    size = bhp.shape[0]

    pred_debit = np.zeros_like(debit, dtype=np.float64)
    pred_cont_pres = np.zeros_like(debit, dtype=np.float64)
    dtau = np.zeros((size, n_wells), dtype=np.float64)
    dalpha = np.zeros((size, n_wells, n_wells), dtype=np.float64)
    domega = np.zeros((size, n_wells, n_wells), dtype=np.float64)
    db = np.zeros((size, n_wells, 1), dtype=np.float64)

    for wi in prange(n_wells):
        pred_debit[:, wi], pred_cont_pres[:, wi], dtau[:, wi], dalpha[:, wi, :], domega[:, wi, :], db[:, wi, :] = _predict_single_well(wi, tau, alpha, exp_omega, b, recovery_time, bhp, debit, p_initial, is_halt, J)

    return pred_debit, pred_cont_pres, dtau, dalpha, domega, db


@jit('Tuple((f8[:, :], f8[:, :], f8[:, :], f8[:, :, :], f8[:, :, :], f8[:, :, :]))(i8, f8[:], f8[:, :], f8[:, :, :], f8[:], f8, f8[:, :], f8[:, :], f8[:], i8[:, :], f8[:])', nopython=True, parallel=True)
def _predict_prod(prod_cnt, tau, alpha, exp_omega, b, recovery_time, bhp, debit, p_initial, is_halt, J):
    n_wells = tau.size
    size = bhp.shape[0]

    pred_debit = np.zeros((size, prod_cnt), dtype=np.float64)
    pred_cont_pres = np.zeros((size, prod_cnt), dtype=np.float64)
    dtau = np.zeros((size, prod_cnt), dtype=np.float64)
    dalpha = np.zeros((size, prod_cnt, n_wells), dtype=np.float64)
    domega = np.zeros((size, prod_cnt, n_wells), dtype=np.float64)
    db = np.zeros((size, prod_cnt, 1), dtype=np.float64)

    for wi in prange(prod_cnt):
        pred_debit[:, wi], pred_cont_pres[:, wi], dtau[:, wi], dalpha[:, wi, :], domega[:, wi, :], db[:, wi, :] = _predict_single_well(wi, tau, alpha, exp_omega, b, recovery_time, bhp, debit, p_initial, is_halt, J)

    return pred_debit, pred_cont_pres, dtau, dalpha, domega, db

@jit('Tuple((f8[:, :], f8[:, :]))(f8[:], f8[:, :], f8[:, :, :], f8[:], f8[:], f8[:, :], f8[:, :], i8[:, :], f8, f8[:], f8[:], f8[:, :])', nopython=True)
def _predict_prod_test(tau, alpha, exp_omega_all, b, prod_0, injection, bhp, is_halt_all, recovery_time, p_initial, J, hist_debit):
    prod_cnt = prod_0.size
    t_size, _ = bhp.shape

    # Пластовое давление
    cont_pres = np.zeros((t_size, prod_cnt))
    cont_pres[0] = p_initial

    # Предсказанный дебит
    pred_production = np.zeros((t_size, prod_cnt))
    pred_production[0] = prod_0
    assert len(is_halt_all) == len(hist_debit) + t_size - 1

    exp_tau = np.exp(-1. / tau)

    for n in range(1, t_size):
        debit = np.hstack((pred_production, -injection))
        for wi in range(prod_cnt):
            work_time = count_working_days(is_halt_all, wi, (len(hist_debit) - 1) + n)
            # предполагается, что на тесте все скважины включены
            # если не работает с самого начала то ничего не делаем, скважина еще не включилась
            # if np.all(is_halt[:n + 1, wi]):
            #     cont_pres[n, wi] = cont_pres[n - 1, wi]
            #     pred_production[n, wi] = 0.
            #     continue

            # Расчет вклада соседей
            Q_n_1 = Q_n(wi, alpha, exp_omega_all, np.vstack((hist_debit[:-1], debit)), (len(hist_debit) - 1) + n - 1)[0]

            # Чекалюк
            if is_halt_all[(len(hist_debit) - 1) + n, wi] or work_time == -1 or work_time > recovery_time:
                Jwi = J[wi]
            else:
                Jwi = J[wi] / (1 + b[0] * np.log(work_time / recovery_time))

            # Обновление пластового давления и дебита
            if is_halt_all[(len(hist_debit) - 1) + n, wi]:
                cont_pres[n, wi] = cont_pres[n - 1, wi] + Q_n_1 / (Jwi * tau[wi])
                pred_production[n, wi] = 0.
            else:
                cont_pres[n, wi] = exp_tau[wi] * cont_pres[n - 1, wi] + (bhp[n, wi] + Q_n_1 / Jwi) * (1 - exp_tau[wi])
                pred_production[n, wi] = Jwi * (cont_pres[n, wi] - bhp[n, wi])

    return pred_production, cont_pres


class OmegaCrm(BaseModel):

    def _init_array(self, val, shape):
        if type(val) is np.ndarray and val.shape == shape:
            return val
        elif type(val) is int or type(val) is float:
            ret = np.ones(shape, dtype=np.float64) * val
            if len(shape) == 2:
                np.fill_diagonal(ret, 0.0)
            return ret
        else:
            raise ValueError("Unsupported attribute type.")

    def __init__(self, coordinates, bnds, J=0.8, tau=100., alpha='dist', b=1.7, omega=1., p_initial=300.,
                 recov_time=5., influence_r=2 * 1e3):
        self.bnds = bnds
        # coordinates initialization
        self.coordinates = coordinates
        n_wells = len(coordinates)
        self.n_wells = n_wells
        self.well_names = coordinates.index.tolist()
        self.recov_time = float(recov_time)

        # initial reservoir pressure init
        self.p_initial = self._init_array(p_initial, (n_wells,))

        # productivity initialization
        self.J = self._init_array(J, (n_wells,))

        # tau initialization
        self.tau = self._init_array(tau, (n_wells,))

        # b initialization
        if b > 1. / np.log(recov_time):
            logger.error('Invalid value for b')
            raise ValueError('Invalid value for b')
        self.b = self._init_array(b, (1,))

        # omega initialization
        self.omega = self._init_array(omega, (n_wells, n_wells))
        assert np.all(np.diag(self.omega) == np.zeros(n_wells))

        self.influence_r = influence_r

        # alpha initialization
        if type(alpha) is str and alpha == 'dist':
            self.alpha = inverse_distance_mtrx(self.coordinates, self.coordinates, dist_thres=influence_r)
            assert self.alpha.shape == (n_wells, n_wells)
            assert np.all(np.diag(self.alpha) == np.zeros(n_wells))
        else:
            self.alpha = self._init_array(alpha, (n_wells, n_wells))

    def pack_params(self):
        assert np.allclose(self.omega, self.omega.T, atol=1e-8)

        csr_alpha = csr_matrix(self.alpha)
        X, Y, alpha_data = find(csr_alpha)
        self.alpha_rsc = ReverseSparseConverter((X, Y), csr_alpha.get_shape())

        # omega and alpha indices must correspond
        csr_omega = csr_matrix(np.triu(self.omega))
        X, Y, omega_data = find(csr_omega)
        self.omega_rsc = ReverseSparseConverter((X, Y), csr_omega.get_shape())

        self.alpha_size, self.omega_size = alpha_data.size, omega_data.size
        self.pl = ParamsLinker(self.tau, alpha_data, omega_data, self.b)
        return self.pl.link(self.tau, alpha_data, omega_data, self.b)

    def _check_for_nans(self, *args):
        for arg in args:
            assert not np.any(np.isnan(arg)), f'Nans in {arg}'

    def unpack_params(self, x):
        tau, alpha_data, omega_data, b = self.pl.split(x)

        alpha = self.alpha_rsc.convert(alpha_data)
        assert np.all(np.diag(alpha) == np.zeros(self.n_wells))

        omega = self.omega_rsc.convert(omega_data)
        omega = omega + omega.T
        assert np.all(np.diag(omega) == np.zeros(self.n_wells))

        return tau, alpha, omega, b

    def alpha_sum_constraint(self, x):
        _, alpha, _, _ = self.unpack_params(x)
        return np.min(1 - np.sum(alpha, axis=1))

    def _get_exp_omega(self, omega, n):
        with np.errstate(divide='ignore', invalid='ignore'):
            exp_omega = np.exp(- omega[:, :, np.newaxis] / np.arange(n)[np.newaxis, np.newaxis, :])
            exp_omega[:, :, 0] = 0.0
        return exp_omega

    def loss_func(self, x, debit, bhp, is_halt):
        """
        Function is necessary for changing loss function in inherited classes.
        """
        size, n_wells = debit.shape
        tau, alpha, omega, b = self.unpack_params(x)
        exp_omega = self._get_exp_omega(omega, debit.shape[0])

        pred_debit, _, dtau, dalpha, domega, db = _predict(tau, alpha, exp_omega, b,
                                                            self.recov_time, bhp, debit, self.p_initial, is_halt, self.J)

        dtau = 2 * np.sum((pred_debit - debit) * dtau, axis=0) / (size * n_wells)
        dalpha = 2 * np.sum((pred_debit - debit)[:, :, np.newaxis] * dalpha, axis=0) / (size * n_wells)
        domega = 2 * np.sum((pred_debit - debit)[:, :, np.newaxis] * domega, axis=0) / (size * n_wells)
        db = 2 * np.sum((pred_debit - debit)[:, :, np.newaxis] * db, axis=(0, 1)) / (size * n_wells)
        domega += domega.T

        return np.mean((pred_debit - debit) ** 2), self.pl.link(dtau, dalpha[self.alpha_rsc.indices], domega[self.omega_rsc.indices], db)

    def fit(self, debit, bhp, minimizer: BaseMinimizer):
        is_halt = (debit == 0).astype(np.int64)
        assert debit.shape == bhp.shape

        x_0 = self.pack_params()

        loss_func = lambda x: self.loss_func(x, debit, bhp, is_halt)

        bounds = ([self.bnds['tau_bnd']] * self.tau.size +
                  [self.bnds['alpha_bnd']] * self.alpha_size +
                  [self.bnds['omega_bnd']] * self.omega_size +
                  [self.bnds['b_bnd']] * self.b.size)
        minimizer.set_bounds(bounds)

        self.x_opt = minimizer.minimize(loss_func, True, x_0)
        self.tau, self.alpha, self.omega, self.b = self.unpack_params(self.x_opt)

        return self

    def predict(self, debit, bhp):
        is_halt = (debit == 0).astype(np.int64)
        exp_omega = self._get_exp_omega(self.omega, debit.shape[0])

        pred_debit, pred_contour_pres = _predict(self.tau, self.alpha, exp_omega, self.b, self.recov_time, bhp, debit,
                                                    self.p_initial, is_halt, self.J)[:2]

        return pred_debit, pred_contour_pres

    def to_excel(self, file_path='OmegaCRM parameters.xlsx'):
        with ExcelExporter(file_path) as excel_exporter:
            excel_exporter.write_series_to_worksheet('tau', self.well_names, self.tau, ['Well', 'tau'])
            excel_exporter.write_series_to_worksheet('J', self.well_names, self.J, ['Well', 'J'])
            excel_exporter.write_series_to_worksheet('b', ['b'], self.b)
            excel_exporter.write_matrix_to_worksheet('omega', self.well_names, self.omega, self.well_names)
            excel_exporter.write_matrix_to_worksheet('alpha', self.well_names, self.alpha, self.well_names)




class OmegaCrmProdOnly(OmegaCrm):

    def __init__(self, prod_cnt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prod_cnt = prod_cnt
        self.prod_well_names = self.well_names[:prod_cnt]
        self.inj_well_names = self.well_names[prod_cnt:]

    def pack_params(self):
        assert np.allclose(self.omega, self.omega.T, atol=1e-8)

        alpha_copy = self.alpha.copy()
        alpha_copy[self.prod_cnt:] = .0
        csr_alpha = csr_matrix(alpha_copy)
        X, Y, alpha_data = find(csr_alpha)
        self.alpha_rsc = ReverseSparseConverter((X, Y), csr_alpha.get_shape())

        # omega and alpha indices must correspond
        triu_omega = np.triu(self.omega)
        triu_omega[self.prod_cnt:] = .0
        csr_omega = csr_matrix(triu_omega)
        X, Y, omega_data = find(csr_omega)
        self.omega_rsc = ReverseSparseConverter((X, Y), csr_omega.get_shape())

        self.alpha_size, self.omega_size = alpha_data.size, omega_data.size
        self.pl = ParamsLinker(self.tau[:self.prod_cnt], alpha_data, omega_data, self.b)
        return self.pl.link(self.tau[:self.prod_cnt], alpha_data, omega_data, self.b)

    def unpack_params(self, x):
        prod_tau, alpha, omega, b = super().unpack_params(x)
        tau = self.tau.copy()
        tau[:self.prod_cnt] = prod_tau
        return tau, alpha, omega, b

    def loss_func(self, x, debit, bhp, is_halt):
        """
        Function is necessary for changing loss function in inherited classes.
        """
        size, n_wells = debit.shape
        tau, alpha, omega, b = self.unpack_params(x)
        exp_omega = self._get_exp_omega(omega, debit.shape[0])

        pred_debit, _, dtau, dalpha, domega, db = _predict_prod(self.prod_cnt, tau, alpha, exp_omega, b,
                                                                self.recov_time, bhp, debit, self.p_initial, is_halt, self.J)

        dtau = 2 * np.sum((pred_debit - debit[:, :self.prod_cnt]) * dtau, axis=0) / (size * self.prod_cnt)
        dalpha = 2 * np.sum((pred_debit - debit[:, :self.prod_cnt])[:, :, np.newaxis] * dalpha, axis=0) / (size * self.prod_cnt)
        domega = 2 * np.sum((pred_debit - debit[:, :self.prod_cnt])[:, :, np.newaxis] * domega, axis=0) / (size * self.prod_cnt)
        db = 2 * np.sum((pred_debit - debit[:, :self.prod_cnt])[:, :, np.newaxis] * db, axis=(0, 1)) / (size * self.prod_cnt)
        domega[:, :self.prod_cnt] += domega[:, :self.prod_cnt].T

        return np.mean((pred_debit - debit[:, :self.prod_cnt]) ** 2), self.pl.link(dtau[:self.prod_cnt], dalpha[self.alpha_rsc.indices], domega[self.omega_rsc.indices], db)

    def fit(self, debit, bhp, minimizer: BaseMinimizer):
        is_halt = (debit == 0).astype(np.int64)
        assert debit.shape == bhp.shape

        x_0 = self.pack_params()

        loss_func = lambda x: self.loss_func(x, debit, bhp, is_halt)

        bounds = ([self.bnds['tau_bnd']] * self.prod_cnt +
                  [self.bnds['alpha_bnd']] * self.alpha_size +
                  [self.bnds['omega_bnd']] * self.omega_size +
                  [self.bnds['b_bnd']] * self.b.size)
        minimizer.set_bounds(bounds)

        self.x_opt = minimizer.minimize(loss_func, True, x_0)
        self.tau, self.alpha, self.omega, self.b = self.unpack_params(self.x_opt)

        return self

    def predict_train(self, debit, bhp):
        is_halt = (debit == 0).astype(np.int64)
        exp_omega = self._get_exp_omega(self.omega, debit.shape[0])

        pred_debit, pred_contour_pres = _predict_prod(self.prod_cnt, self.tau, self.alpha, exp_omega, self.b, self.recov_time, bhp, debit,
                                                        self.p_initial, is_halt, self.J)[:2]

        return pred_debit, pred_contour_pres

    def predict_test(self, bhp, injection, debit_train, prod_0, p_initial, is_halt_all):
        pred_debit_test, contour_pres_test = _predict_prod_test(self.tau,
                                                                self.alpha,
                                                                self._get_exp_omega(self.omega, is_halt_all.shape[0]),
                                                                self.b, prod_0,
                                                                injection, bhp, is_halt_all, self.recov_time,
                                                                p_initial, self.J, debit_train)


        return pred_debit_test, contour_pres_test

    def to_excel(self, file_path='OmegaCRM(ProdOnly) parameters.xlsx'):
        with ExcelExporter(file_path,  {'nan_inf_to_errors': True}) as excel_exporter:
            excel_exporter.write_series_to_worksheet('tau', self.prod_well_names, self.tau[:self.prod_cnt], ['Well', 'tau'])
            excel_exporter.write_series_to_worksheet('J', self.prod_well_names, self.J[:self.prod_cnt], ['Well', 'J'])
            excel_exporter.write_series_to_worksheet('b', ['b'], self.b)
            excel_exporter.write_matrix_to_worksheet('omega', self.prod_well_names, self.omega[:self.prod_cnt], self.well_names)
            excel_exporter.write_matrix_to_worksheet('alpha', self.well_names, self.alpha[:self.prod_cnt].T, self.prod_well_names)