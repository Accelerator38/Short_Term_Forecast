class ConfigCRM:
    def __init__(self,
                 date_start_adapt,
                 date_end_adapt,
                 date_end_forecast,
                 calc_CRM=True,
                 calc_CRMIP=False,
                 grad_format_data=True,
                 oilfield='Отдельное'):
        self.date_start_adapt = date_start_adapt
        self.date_end_adapt = date_end_adapt
        self.date_end_forecast = date_end_forecast
        self.calc_CRM = calc_CRM
        self.calc_CRMIP = calc_CRMIP
        self.grad_format_data = grad_format_data
        self.oilfield = oilfield
        self.CUT_DATE = self.date_start_adapt
        self.CUT_DATE2 = self.date_end_adapt

        # схема модели svir/lwir
        self.SCHEME = 'LVIR'  # 'SVIR'
        # начальное значение параметров модели
        self.TAU_0 = 50
        self.J_0 = 10
        # радиус влияния (вне его скважины считаются невзаимодействующими)
        self.INFLUENCE_R = 1300
        # учёт радиуса от хвоста и пятки скважины
        # TODO: заменить на True, когда будут данные по хвосту/пятке скважины
        self.double_R = False
        self.inactive_time = 0.8
        # какие модели строить
        self.liquid_flag = self.wc_flag = self.gas_factor_flag = self.pressure_reservoir_flag = True
        # Параметры оптимизации SLSQP
        self.options_SLSQP_CRM = {'maxiter': 100, 'ftol': 1e-7, 'eps': 1.49e-8, 'disp': True}
        # CRM, CRMIP
        # Ограничения на оптимизируемые параметры для CRM/CRMIP
        self.prod_bnd = (0, 500)
        self.tau_bnd = (1e-6, 3000)
        self.J_bnd = (1e-6, 10)
        self.f_bnd = (0, 1)
        self.bnds_CRM = {'prod_bnd': self.prod_bnd, 'tau_bnd': self.tau_bnd, 'J_bnd': self.J_bnd, 'f_bnd': self.f_bnd}
        # Функция востановления давления
        self.p_res = 220  # пластовое давление
        self.p_0 = 10
        self.recov_time = 100  # характерное время восстановления давления
