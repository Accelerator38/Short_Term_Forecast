import multiprocessing

from loguru import logger

from frameworks_ftor.ftor._wrapper_well import WrapperWell
from frameworks_ftor.ftor.config import Config
from frameworks_ftor.ftor.well import Well


class Calculator(object):

    def __init__(
            self,
            config: Config,
            wells: list[Well],
            n_workers=None,
            logging: bool = False,
    ):
        self._config = config
        self._wells_input = wells
        self._n_workers = n_workers
        self._logging = logging

        self._wells_output = []
        self._kwargs = {}
        if self._logging:
            from logs.worker import Worker
            self._worker = Worker()
        self._run()

    def _run(self) -> None:
        kwargs = self._get_pool_kwargs()
        with multiprocessing.Pool(self._n_workers, **kwargs) as pool:
            self._wells_output = pool.map(self._compute_well, self._wells_input)

    def _compute_well(self, well_input: Well) -> Well:
        self._print_funcs = self._get_print_funcs()
        try:
            self._print_funcs['info'](f'Ftor: well {well_input.well_name}')
            well_output = WrapperWell(self._config, well_input).well
            self._print_funcs['success'](f'Ftor: success {well_input.well_name}')
            return well_output
        except Exception as exc:
            self._print_funcs['exception'](f'Ftor: FAIL {well_input.well_name}', exc)

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
    def wells(self) -> list[Well]:
        return [well for well in self._wells_output if well is not None]
