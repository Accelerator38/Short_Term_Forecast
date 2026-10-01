class Config:
    """Конфигурация гиперпараметров для байесовского ансамбля моделей.

    adaptation_days_number: int, default=30
        Количество дней адаптации во входных данных.

    interval_probability: float, default=0.9
        Уровень значимости при построения доверительного интервала.

    draws: int, default=300
        Количество сэмплов для расчета внутри модели.

    tune: int, default=200
        Количество итераций для настройки во время расчета модели.

    chains: int, default=1
        Количество строимых цепей. Запуск независимых цепей важен для
        некоторых статистик сходимости и увеличивает вероятность корректно
        описать апостериорные распределения с несколькими вершинами.

    cores: int, default=1
        Число цепей, запускаемых параллельно.

    target_accept: float in [0, 1], default=0.95
        Уровень принятия шага. Более высокие значения, такие как 0.9 или 0.95,
        часто лучше работают для проблемных апостериорных распределений.

    name_of_y_true: str
        Название столбца с фактическими данными
    """

    def __init__(
            self,
            adaptation_days_number: int = 30,
            interval_probability: float = 0.9,
            draws: int = 300,
            tune: int = 200,
            chains: int = 1,
            cores: int = 1,
            target_accept: float = 0.95,
            name_of_y_true: str = 'y_true'
    ):
        self.adaptation_days_number = adaptation_days_number
        self.interval_probability = interval_probability
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.cores = cores
        self.target_accept = target_accept
        self.name_of_y_true = name_of_y_true
