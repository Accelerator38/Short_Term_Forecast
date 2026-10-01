class Config(object):
    def __init__(
            self,
            use_equal_time_algorithm: bool = True,
            apply_last_points_adaptation: bool = True,
            number_of_last_points: int = 5,
            param_name_last_points_adaptation: str = 'pressure_initial',
            param_bounds_last_points_adaptation: None or list[float] = None,
    ):

        if param_bounds_last_points_adaptation is None:
            param_bounds_last_points_adaptation = [20, 1000]
        self.use_equal_time_algorithm = use_equal_time_algorithm
        self.apply_last_points_adaptation = apply_last_points_adaptation
        self.number_of_last_points = number_of_last_points
        self.param_name_last_points_adaptation = param_name_last_points_adaptation
        self.param_bounds_last_points_adaptation = param_bounds_last_points_adaptation
