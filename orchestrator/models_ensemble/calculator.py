import multiprocessing

import pandas as pd
from loguru import logger

from models_ensemble.bayesian_model import BayesianModel
from models_ensemble.config import Config


class Calculator:
    def __init__(
            self,
            config: Config,
            data: list[dict],
            n_workers: int = None,
            logging: bool = False,
    ):
        self._config = config
        self._data = data
        self._n_workers = n_workers
        self._logging = logging

        self._output = []
        self._kwargs = {}
        if self._logging:
            from logs.worker import Worker
            self._worker = Worker()
        self._run_multiprocess()

    def _run_multiprocess(self) -> None:
        kwargs = self._get_pool_kwargs()
        with multiprocessing.Pool(self._n_workers, **kwargs) as pool:
            self._output = pool.map(self._run_ensemble, self._data)

    def _run_ensemble(self, well_input: dict) -> dict or None:
        self._print_funcs = self._get_print_funcs()
        try:
            self._print_funcs['info'](f'Ensemble: well {well_input["wellname"]}')
            ensemble = BayesianModel(self._config, well_input['df'])
            result = {'result_train': ensemble.result_train,
                      'result_test': ensemble.result_test,
                      'weights': ensemble.weights,
                      'wellname': well_input['wellname']}
            self._print_funcs['success'](f'Ensemble: success {well_input["wellname"]}')
            return result
        except Exception as exc:
            self._print_funcs['exception'](f'Ensemble: FAIL {well_input["wellname"]}', exc)

    def _get_pool_kwargs(self) -> dict:
        if self._logging:
            return {'initializer': self._worker.set_logger, 'initargs': (logger,)}
        return {}

    def _get_print_funcs(self) -> dict[str, callable]:
        """Определяет использовать обычный print() или логировать.

        В случае logging=True возвращает необходимые методы logger'а, иначе print.
        """

        print_funcs = {'info': print, 'success': print, 'exception': print}
        if self._logging:
            print_funcs = {'info': self._worker.logger.info,
                           'success': self._worker.logger.success,
                           'exception': self._worker.logger.exception}
        return print_funcs

    @property
    def result_train(self) -> dict[str, pd.DataFrame]:
        return {res['wellname']: res['result_train'] for res in self._output if res is not None}

    @property
    def result_test(self) -> dict[str, pd.DataFrame]:
        return {res['wellname']: res['result_test'] for res in self._output if res is not None}

    @property
    def weights(self) -> dict[str, pd.DataFrame]:
        return {res['wellname']: res['weights'] for res in self._output if res is not None}
