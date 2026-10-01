from abc import ABCMeta, abstractmethod
from logging import getLogger
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

logger = getLogger(__name__)


class BaseMinimizer(metaclass=ABCMeta):  # оптимизатор, задание ограничений и параметров

    def __init__(self, options, verbose=True):
        self.constraints = []
        self.bounds = []

        self.options = options
        self.verbose = verbose

    def set_constraints(self, constraints: list):  # установка ограничений
        self.constraints = constraints
        return self

    def set_bounds(self, bounds: list):  # установка граничных условий
        self.bounds = bounds
        return self

    @abstractmethod
    def minimize(self, loss_func: Callable, jac: bool, x_0: np.ndarray):  # минимизатор
        pass


class LogErrorPlotter():  # класс ошибок графиков (почему много закоменчено?)

    def __init__(self, loss_func, jac):
        self.iter_n = 0
        self.err = (lambda x: loss_func(x)[0]) if jac else loss_func
        self.err_vals = []
        plt.ion()
        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(111)
        # self.log_err_line, = self.ax.plot([], [], color='r', marker='+')
        # self.hline = self.ax.axhline(ls='--')
        logger.info('Optimization was started.')

    def __call__(self, x):
        self.iter_n += 1
        self.err_vals.append(np.log(self.err(x)))
        # self.log_err_line.set_data(range(self.iter_n), self.err_vals)
        # self.hline.set_ydata(np.min(self.err_vals))
        # self.ax.set_title('Log loss: {} after {} iterations'.format(np.min(self.err_vals), self.iter_n))
        # self.ax.relim()
        # self.ax.autoscale_view()
        # plt.draw()
        # plt.pause(1e-6)

        logger.info('Current loss: {:.3f} | Lowest loss: {:.3f} | Iteration: {}'
                    .format(self.err_vals[-1], np.min(self.err_vals), self.iter_n))


class SciPySlsqpWrapper(BaseMinimizer):  # минимизатор
    def minimize(self, loss_func, jac, x_0):
        res = minimize(
            loss_func,
            x_0,
            method='SLSQP',
            jac=jac,
            bounds=self.bounds,
            constraints=self.constraints,
            options=self.options,
            callback=LogErrorPlotter(loss_func, jac) if self.verbose else None
        )
        # if self.verbose:
        #     plt.ioff()
        #    plt.show()
        return res.x


# !TODO refactor
class ClustSLSQPMinimizer(BaseMinimizer):  # кластовый минимизатор, поностью переписать

    def __init__(self, constraints, options, verbose, clust_seq, x_mask_getter, epochs_num):
        raise NotImplemented('TODO')
        # super().__init__(constraints, options, verbose)
        # # Cluster zero for parametrs which are shared between clusters
        # self.clust_seq = np.append(np.array(clust_seq), 0)
        # self.x_mask_getter = x_mask_getter
        # self.epochs_num = epochs_num

    def minimize(self, loss_func, x_0, bounds, pen_coef=10):
        raise NotImplemented('TODO')
        # logger.info("Minimization process started...")
        # x = x_0
        # clbk = LogErrorPlotter(loss_func) if self.verbose else None
        # for epoch in range(self.epochs_num):
        #     logger.info(f"\tEPOCH #{epoch}")
        #     for cluster_n in self.clust_seq:
        #         mask = self.x_mask_getter(cluster_n)
        #         if np.sum(mask) == 0:
        #             logger.info(f"\t\tCLUSTER #{cluster_n} | NUM OF OPT PARAMS = 0 | SKIPPING")
        #             continue
        #         assert mask.size == x.size
        #
        #         def clust_loss(x_clust):
        #             x[mask] = x_clust
        #             penalty = 0
        #             for s in self.constraints:
        #                 penalty += np.sum(np.min(s['fun'](x), 0) ** 2)
        #             return loss_func(x) + pen_coef * penalty
        #
        #         if clbk is not None:
        #             clbk.err = clust_loss
        #
        #         res = minimize(
        #             clust_loss,
        #             x[mask],
        #             method='SLSQP',
        #             bounds=[b for m, b in zip(mask, bounds) if m],
        #             options=self.options,
        #             callback=clbk
        #         )
        #         logger.info(f"\t\tCLUSTER #{cluster_n} | NUM OF OPT PARAMS = {np.sum(mask)} | LOGLOSS = {np.log(clust_loss(res.x)):.3}")
        #         x[mask] = res.x
        # if self.verbose:
        #     plt.ioff()
        #     plt.show()
        # return x
