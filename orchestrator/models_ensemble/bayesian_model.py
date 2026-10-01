from datetime import timedelta

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
from theano import shared

from models_ensemble.config import Config


class BayesianModel:
    """Метод Монте-Карло на Марковских цепях. Линейная Байесовская модель.

    Данная модель предполагает, что фактические измерения есть линейная
    комбинация прогнозов по каждой из модели. Внутри модели используется
    алгоритм Метрополиса для построения Марковских цепей.

    Перед использованием модели необходимо задать априорные вероятностные
    распределения весов `w`. Далее модель адаптируется на фактические
    измерения, в итоге для каждого веса `w` получается апостериорное
    распределение. Для каждого из них берется среднее значение и принимается
    за вес соответствующей модели.

    Parameters
    ----------
    config: Config
        конфигурация гиперпараметров модели
    df: pd.DataFrame
        Входные данные. Индекс - это данные по оси x, произвольное
        количество столбцов - это прогнозы по моделям **y_model**,
        еще один столбец с фактическими измерениями **y_true**.

    Notes
    -----
    После адаптации модели происходит вычисление отклонения прогноза от
    последних 5 фактических измерений. Данный сдвиг `delta` будет добавлен к
    прогнозу ансамблевой модели и на прогнозе тоже. Таким образом уменьшается
    ошибка на начальном периоде прогноза.

    Для построения доверительного интервала вокруг прогноза модели используется
    метод семплирования.

    References
    ----------
    .. [1] https://docs.pymc.io/pymc-examples/examples/getting_started.html

    .. [2] https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo

    .. [3] https://habr.com/ru/company/wunderfund/blog/279545/

    """

    def __init__(self,
                 config: Config,
                 df: pd.DataFrame):
        self._adaptation_days_number = config.adaptation_days_number
        self._interval_probability = config.interval_probability
        self._draws = config.draws
        self._tune = config.tune
        self._chains = config.chains
        self._cores = config.cores
        self._target_accept = config.target_accept
        self._name_of_y_true = config.name_of_y_true
        self._df = df[:]
        self._run()

    def _run(self) -> None:
        self._prepare_data()
        self._fit()
        self._extract_weights()
        self._calculate_delta()
        self._predict()
        self._extract_ensemble_forecast(alpha=self._interval_probability)

    def _prepare_data(self) -> None:
        """Подготовка данных для расчетов."""

        # Метод последовательно выполняет следующее:
        # - Удаление строк, где скважина не работала
        # - Удаление строк, где есть NaN
        # - Выделение имен моделей на входе
        # - Создание датасетов адаптации, прогноза
        self._date_test = self._df.index[0] + timedelta(days=self._adaptation_days_number)
        self._df = self._df[self._df[self._name_of_y_true] != 0]
        self._df = self._df.dropna(axis=0, how='any')
        if self._df.empty:
            raise ValueError('Ensemble input has no valid rows after zero/NaN filtering.')

        self._models = list(self._df.columns.values)
        self._models.remove(self._name_of_y_true)
        if not self._models:
            raise ValueError('Ensemble input must contain at least one model forecast column.')

        self._df_train = self._df[:self._date_test]
        self._df_test = self._df[self._date_test:]
        if self._df_train.empty or self._df_test.empty:
            raise ValueError('Ensemble input must contain non-empty train and forecast splits.')

    def _fit(self) -> None:
        """Адаптирует веса моделей на фактические данные."""

        # Метод последовательно выполняет следующее:
        # - Конвертирует данные к типу `shared()`
        # - Задает линейную модель с предположениями об априорных вероятностных распределениях весов
        # - Вычисляет распределения весов моделей, адаптируясь на фактические измерения
        # - Сохраняет рассчитанную модель в `model_g`, `trace_g` для дальнейших расчетов
        self._convert_data_to_shared(self._df_train)
        with pm.Model() as model_g:
            w_const = pm.Normal(name='w_const', mu=0, sd=1)
            w_sd1 = pm.HalfNormal(name='w_sd1', sd=80)
            w_sd2 = pm.HalfNormal(name='w_sd2', sd=30)
            linear_regression = w_const
            for model_name in self._models:
                linear_regression += pm.HalfNormal(name=f'w_{model_name}', sd=1) * self._y_models_shared[model_name]
            mu = pm.Deterministic('mu', linear_regression)
            sd = pm.Deterministic('sd', w_sd1 + w_sd2 * self._x_delta)
            y_ensemble = pm.Normal(name='y_pred', mu=mu, sd=sd, observed=self._y_true_shared)
            trace_g = pm.sample(
                draws=self._draws,
                chains=self._chains,
                tune=self._tune,
                target_accept=self._target_accept,
                cores=self._cores,
                random_seed=1,
                return_inferencedata=False
            )
        self._model_g = model_g
        self._trace_g = trace_g

    def _predict(self) -> None:
        """Семплирование ансамбля на прогнозе."""

        self._set_values_to_shared_data(self._df_test)
        # - Генерация семплов из построенной модели для построения доверительного интервала на прогнозе
        self._ppc = pm.sample_posterior_predictive(self._trace_g, model=self._model_g)

    def _convert_data_to_shared(self, data: pd.DataFrame) -> None:
        """Конвертирование входных данных к формату `shared`."""

        # - Выделяются следующие ряды: прогнозы моделей, фактические значения и массив `_x_delta`, отвечающий за
        #   увеличение ошибки по мере отдаления от момента конца адаптации `_x_adapt_end`.
        # - Библиотека pymc3 требует, чтобы переменные были обернуты в `shared()`.
        self._x = pd.to_datetime(data.index.to_series())
        self._x_adapt_end = self._x[-1]
        time_deltas = np.array([abs((elem - self._x_adapt_end).days) for elem in self._x])
        self._x_delta = shared(time_deltas)
        self._y_true = data[self._name_of_y_true].values
        self._y_true_shared = shared(self._y_true)

        self._y_models_shared = {}
        for model_name in self._models:
            y_model_forecast = data[model_name].values
            self._y_models_shared[model_name] = shared(y_model_forecast)

    def _set_values_to_shared_data(self, data: pd.DataFrame) -> None:
        """Устанавливает новые значения в переменные типа `shared`."""

        # - Аналог функции _convert_data_to_shared.
        # - Перезаписываются следующие ряды: прогнозы моделей, фактические значения и массив `_x_delta`.
        # - Чтобы данные перезаписались, недостаточно переприсвоить переменным новые значения: `val = shared(new val)`.
        #   Необходимо использовать метод set_value() для уже имеющихся shared-переменных: `val.set_value(new_val)`.
        self._x = pd.to_datetime(data.index.to_series())
        time_deltas = np.array([abs((elem - self._x_adapt_end).days) for elem in self._x])
        self._x_delta.set_value(time_deltas)
        y_true = data[self._name_of_y_true].values
        self._y_true_shared.set_value(y_true)
        for model_name in self._models:
            y_model_forecast = data[model_name].values
            self._y_models_shared[model_name].set_value(y_model_forecast)

    def _extract_weights(self) -> None:
        """Вычисление среднего для каждого из распределений весов."""

        # Из полученных распределений весов находит среднее `.mean()` для каждого из них - эти значения принимаются
        # за веса моделей для построения прогноза ансамбля
        const = self._trace_g['w_const'].mean()
        sd1 = self._trace_g['w_sd1'].mean()
        sd2 = self._trace_g['w_sd2'].mean()
        self._weights = {
            'const': round(const, 2),
            'sd1': round(sd1, 2),
            'sd2': round(sd2, 2),
        }
        for model_name in self._models:
            model_weight = self._trace_g[f'w_{model_name}'].mean()
            self._weights[model_name] = round(model_weight, 2)

    def _calculate_delta(self) -> None:
        """Вычисление сдвига прогноза ансамбля под последние 5 точек факта."""

        # Краткий вид формулы:
        #   delta = y_true - (w_const + w_1 * forecast_1 + ... + w_n * forecast_n)
        diff = self._y_true[-5:]
        diff = diff - self._weights['const'] * np.ones(len(diff))
        for model_name in self._models:
            diff = diff - self._weights[model_name] * self._df[model_name].values[-5:]
        delta = diff.mean()
        self._delta = delta

    def _extract_ensemble_forecast(self, alpha=0.9) -> None:
        """Извлечение прогноза и доверительных интервалов из семплирования."""

        # Из рассчитанных цепей ансамбля извлекаются:
        #     1) доверительный интервал с заданным уровнем доверия alpha
        #     2) прогноз ансамбля = delta + (w_const + w_1 * forecast_1 + ... + w_n * forecast_n)
        y_pred_samples = np.asarray(self._ppc['y_pred'] + self._delta)
        if y_pred_samples.ndim == 2:
            y_pred_samples = y_pred_samples[np.newaxis, :, :]
        hdi = az.hdi(y_pred_samples, hdi_prob=alpha)
        confidence_interval_lower = hdi[:, 0]
        confidence_interval_upper = hdi[:, 1]
        if np.alltrue(confidence_interval_lower >= confidence_interval_upper):
            temp = confidence_interval_lower
            confidence_interval_lower = confidence_interval_upper
            confidence_interval_upper = temp

        ensemble_forecast = self._delta + self._weights['const']
        for model_name in self._models:
            ensemble_forecast += self._weights[model_name] * self._df[model_name].values
        self._df['ensemble'] = ensemble_forecast
        self._df.loc[self._x, 'interval_lower'] = confidence_interval_lower
        self._df.loc[self._x, 'interval_upper'] = confidence_interval_upper
        self._df_result = self._df[:]

    @property
    def result_train(self) -> pd.DataFrame:
        return self._df_result.loc[self._df_train.index, 'ensemble']

    @property
    def result_test(self) -> pd.DataFrame:
        return self._df_result.loc[self._df_test.index, :]

    @property
    def weights(self) -> dict:
        return self._weights
