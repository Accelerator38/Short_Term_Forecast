import numpy as np
from numba.core import types
from numba import njit
from numba.pycc import CC
from numba.typed import Dict, List
import optimize_lestsq as optim
import constants as const
from analytical_model import calc_points

float_array = types.double[:]
cc = CC('analytical_optimize')
cc.output_dir = 'aot'

# find_params возвращает результат поиска коэффициентов для решения задачи поиска минимума функции. Используется
# алгоритм Левеньерга-Маргуардта с модификациями: ограничение поиска в задданом диапазоне, стратегия обновления
# матрицы Гессе, аппроксимация матрицы Якоби.
#
# Входные параметры:
# Dict(int, double) settings - параметры модели в формате ключ(Параметр)-значение. Первое приближение составляется из данного словаря
# double[] times - массив времени фактических значений
# double[] values - массив фактических значений (давление/дебиты), относительно times
# int well - модель скважины
# int wbs - модель ВСС
# int layer - модель пласта
# int bound - модель границы. Если в массиве param_array находится параметр BOUND_TYPE, то первое приближение для
#             данного параметра берется из значения bound, которое подается в функцию
# double[][] rates - массив давлений/дебитов ф формате [[time, value], ...]
# int[] param_array - массив параметров для поиска
# double[][] min_max - массив границ парамтров для поиска ф формате [[min, max], ...]
# boolean is_rate - если is_rate == true, то будет производиться расчет дебитов, иначе расчет давлений
# int max_iter - критейри остановки. Максимальное значение итераций марквардта при решении задачи поиска минимума
# double tau - отвечат за начальное задание лямбдыиспользуется только при lm_implement == LM_IMPLEMENT_WITH_DAMPING_CONTROL
# double e1 - критерий остановки. Максимальная величина градиента. Используется только при lm_implement == LM_IMPLEMENT_WITH_DAMPING_CONTROL
# double e2 - критерий остановки. Максимальное относительное изменение длины вектора искомых параметров.
#                                 Используется только при lm_implement == LM_IMPLEMENT_WITH_DAMPING_CONTROL
# double max_error - критерий остановки. Максимальное квадратичное отклониние
# double e4 - критерий остановки. Максимальное относительное уменьшение величины ошибки.
#             Используется только при lm_implement == LM_IMPLEMENT_WITH_DAMPING_CONTROL
# double delta_error - критерий остановки.  максимальное квадратичное отклониние от предыдущей итерации марквардта
# int bound_strategy - парамтер, отвечающий за способ ограничения поиска в заданных интервалах. Может быть равен:
#                      BOUND_STRATEGY_HELMUS = 0
#                      BOUND_STRATEGY_CYCLE_LEFT_TO_RIGHT = 1
#                      BOUND_STRATEGY_CYCLE_MIRROR = 2
# int lm_implement - параметр, отвечающий за способ реализации алгоритма Левенберга-Марнуардта. Может быть равен:
#                    LM_IMPLEMENT_DEFAULT = 0
#                    LM_IMPLEMENT_WITH_DAMPING_CONTROL = 1
# int broyden_period - параметр, отвечающий за периоодичность замены численного расчета матрицы Якоби аппроксимацией.
#                      каждая broyden_period итерация использует аппроксимацию матрицы Якоби
# int hessian_update_strategy - параметр, отвечающий за способ обновелния матрицы Гессе. Может быть равен:
#                               HESSIAN_UPDATE_STRATEGY_MARQUARDT = 0
#                               HESSIAN_UPDATE_STRATEGY_DEFAULT = 1
#
# Возвращаемый параметр:
# List(double[]) результаты, где list[0] - массив double[] с найденными параметрами, list[1][0] - количсевто итераций
#                                          марквардта, list[1][1] - количество вызовов функции
@njit
@cc.export('find_params', 'ListType(f8[::1])(DictType(i8, f8), f8[:], f8[:], i4, i4, i4, i4, f8[:, :], i4[:], f8[:, :], b1, i4, f8, f8, f8, f8, f8, f8, i4, i4, i4, i4)')
def find_params(settings, times, values, well, wbs, layer, bound, rates, param_array, min_max, is_rate, max_iter,
                tau, e1, e2, max_error, e4, delta_error, bound_strategy, lm_implement, broyden_period, hessian_update_strategy):
    min_max_copy = min_max.copy()
    converted_param = param_array.astype(np.int64)
    dynamic_bound_map = create_dynamic_bound_map()
    validate_min_max(converted_param, min_max_copy, dynamic_bound_map)
    settings_copy = create_copy_settings(settings, bound)
    param_map, param_map_revers = create_param_maps(param_array)
    first = create_first_guess(param_array, settings_copy, param_map)
    args = (times, rates, settings_copy, param_map, param_map_revers, min_max_copy, dynamic_bound_map)
    constants = (well, wbs, layer, bound, 0 if is_rate else 1, bound_strategy)
    if is_rate:
        rates[0][2] = settings.get(const.PRESSURE_INITIAL, const.DEFAULT_PRESSURE_INITIAL)
    if lm_implement == const.LM_IMPLEMENT_WITH_DAMPING_CONTROL:
        iters, fun_calls, sol = optim.solve_lm_damping_control(fun, jac, get_validated_bounds, first, bound_strategy, min_max, max_iter, delta_error, e1, e2, max_error, e4, tau, broyden_period, hessian_update_strategy, values, args=args, constants=constants, display=False)
    else:
        iters, fun_calls, sol = optim.solve_lm_lu_with_bounds(fun, jac, get_validated_bounds, first, bound_strategy, min_max, max_iter, max_error, delta_error, broyden_period, hessian_update_strategy, values, args=args, constants=constants, display=False)
    frac_count_index = param_map_revers.get(const.FRACTURES_COUNT)
    if frac_count_index is not None:
        index = int(frac_count_index)
        sol[index] = float(int(sol[index]))
    bound_index = param_map_revers.get(const.BOUND_TYPE)
    if bound_index is not None:
        index = int(bound_index)
        sol[index] = float(round(sol[index]))
    ans = List()
    ans.append(sol)
    ans.append(np.array([float(iters), float(fun_calls)]))
    return ans


@njit
def fun(params, args, constants):
    variables = args[0]
    rates = args[1]
    settings_copy = args[2]
    param_map = args[3]
    param_map_revers = args[4]
    well = constants[0]
    wbs = constants[1]
    layer = constants[2]
    bound = int(round(settings_copy.get(const.BOUND_TYPE, constants[3])))
    is_rate = int(round(constants[4])) == 0
    set_params(params, settings_copy, param_map, param_map_revers, rates, is_rate)
    ans = np.zeros(len(variables))
    calc_p = calc_points(settings_copy, variables, well, wbs, layer, bound, rates, is_rate=is_rate)
    for i in range(0, len(variables)):
        ans[i] = calc_p[i][1]
    return ans


@njit
def jac(params, args, constants, current_bounds):
    variables = args[0]
    rates = args[1]
    settings_copy = args[2]
    param_map = args[3]
    param_map_revers = args[4]
    well = constants[0]
    wbs = constants[1]
    layer = constants[2]
    is_rate = int(round(constants[4])) == 0
    bound_strategy = constants[5]
    params_ext = optim.a_ext_all_with_strategy(current_bounds, params, bound_strategy)
    ans = np.zeros((len(variables), len(params)))
    for i in range(0, len(params)):
        min_max_cur = current_bounds[i]
        key = param_map.get(i)
        true_value_int = params[i]
        true_value_ext = params_ext[i]
        if bound_strategy == const.BOUND_STRATEGY_HELMUS:
            if key == const.FRACTURES_COUNT:
                d_param = optim.a_int_helmus(min_max_cur, params_ext[i] + 1.5) - true_value_int
            elif key == const.BOUND_TYPE:
                d_param = optim.a_int_helmus(min_max_cur, params_ext[i] + 1.0) - true_value_int
            else:
                d_param = optim.a_int_helmus(min_max_cur, params_ext[i] * 1.01) - true_value_int
        else:
            if key == const.FRACTURES_COUNT:
                d_param = max(1.5, abs(0.01 * true_value_int))
            elif key == const.BOUND_TYPE:
                d_param = 1.0
            else:
                d_param = abs(0.01 * true_value_int)
        if d_param == 0:
            d_param = 0.1
        d_param_lower = d_param
        d_param_upper = d_param
        if params_ext[i] == min_max_cur[0] or params_ext[i] == min_max_cur[1]:
            if key == const.FRACTURES_COUNT or key == const.BOUND_TYPE:
                if bound_strategy == const.BOUND_STRATEGY_HELMUS:
                    d_param_upper = d_param_lower + (optim.a_int_helmus(min_max_cur, params_ext[i] + 1.0) - true_value_int)
                else:
                    d_param_upper = d_param_lower + 1
            else:
                d_param_upper = d_param_lower * 1.2
        upper_delta = true_value_int + d_param_upper
        lower_delta = true_value_int - d_param_lower
        params_ext[i] = optim.a_ext_with_strategy(min_max_cur, upper_delta, bound_strategy)
        set_params(params_ext, settings_copy, param_map, param_map_revers, rates, is_rate)
        bound = int(round(settings_copy.get(const.BOUND_TYPE, constants[3])))
        upper = calc_points(settings_copy, variables, well, wbs, layer, bound, rates, is_rate=is_rate)
        params_ext[i] = optim.a_ext_with_strategy(min_max_cur, lower_delta, bound_strategy)
        set_params(params_ext, settings_copy, param_map, param_map_revers, rates, is_rate)
        bound = int(round(settings_copy.get(const.BOUND_TYPE, constants[3])))
        lower = calc_points(settings_copy, variables, well, wbs, layer, bound, rates, is_rate=is_rate)
        for j in range(0, len(upper)):
            ans[j][i] = (upper[j][1] - lower[j][1]) / (d_param_upper + d_param_lower)
        params_ext[i] = true_value_ext
        set_params(params_ext, settings_copy, param_map, param_map_revers, rates, is_rate)
    return ans


@njit
def set_params(params, settings, param_map, param_map_revers, rates, is_rate):
    has_rect_width = param_map_revers.get(const.RECTANGLE_WIDTH) is not None
    has_rect_length = param_map_revers.get(const.RECTANGLE_LENGTH) is not None
    has_n_frac = param_map_revers.get(const.FRACTURES_COUNT) is not None
    has_bound_type = param_map_revers.get(const.BOUND_TYPE) is not None
    for i in range(0, len(params)):
        settings[param_map.get(i)] = params[i]
    if is_rate:
        rates[0][2] = settings.get(const.PRESSURE_INITIAL)
    if has_n_frac:
        settings[const.FRACTURES_COUNT] = round(settings.get(const.FRACTURES_COUNT))
    if has_bound_type:
        settings[const.BOUND_TYPE] = round(settings.get(const.BOUND_TYPE))
    update_by_rect_w_l(settings, has_rect_width, has_rect_length)
    cfd = 1.0
    settings[const.FRACTURE_CONDUCTIVITY] = cfd * settings[const.PERMEABILITY] * settings[const.FRACTURE_HALF_LENGTH]


@njit
def create_param_maps(param_array):
    param_map = Dict.empty(key_type=types.int64, value_type=types.int64)
    param_map_revers = Dict.empty(key_type=types.int64, value_type=types.int64)
    for i in range(0, len(param_array)):
        param_map[i] = param_array[i]
        param_map_revers[param_array[i]] = i
    return param_map, param_map_revers


@njit
def create_first_guess(param_array, settings_copy, param_map):
    first = np.zeros(len(param_array))
    for i in range(0, len(param_array)):
        first[i] = settings_copy.get(param_map.get(i))
    return first


@njit
def check_is_rate_bound_rect(settings, rates, is_rate, has_rect_width, has_rect_length):
    update_by_rect_w_l(settings, has_rect_width, has_rect_length)
    if is_rate:
        rates[0][2] = settings.get(const.PRESSURE_INITIAL)


@njit
def create_copy_settings(settings_orig, bound):
    settings = Dict.empty(key_type=types.int64, value_type=types.double)
    settings[const.TIME_INITIAL] = const.DEFAULT_TIME_INITIAL
    settings[const.PRESSURE_INITIAL] = const.DEFAULT_PRESSURE_INITIAL
    settings[const.VOLUME_FACTOR] = const.DEFAULT_VOLUME_FACTOR
    settings[const.VISCOSITY] = const.DEFAULT_VISCOSITY
    settings[const.PERMEABILITY] = const.DEFAULT_PERMEABILITY
    settings[const.LAYER_HEIGHT] = const.DEFAULT_LAYER_HEIGHT
    settings[const.POROSITY] = const.DEFAULT_POROSITY
    settings[const.COMPRESSIBILITY_TOTAL] = const.DEFAULT_COMPRESSIBILITY
    settings[const.WELL_RADIUS] = const.DEFAULT_WELL_RADIUS
    settings[const.SKIN] = const.DEFAULT_SKIN
    settings[const.SKIN_CALC] = const.DEFAULT_SKIN_CALC
    settings[const.LAMBDA] = const.DEFAULT_LAMBDA
    settings[const.OMEGA] = const.DEFAULT_OMEGA
    settings[const.MOBILITY_RATIO] = const.DEFAULT_MOBILITY_RATIO
    settings[const.DIFFUSIVITY_RATIO] = const.DEFAULT_DIFFUSIVITY_RATIO
    settings[const.RADIAL_COMPOSITE_RADIUS] = const.DEFAULT_RADIAL_COMPOSITE_RADIUS
    settings[const.BOUND_RADIUS] = const.DEFAULT_BOUND_RADIUS
    settings[const.BOUND_1] = const.DEFAULT_BOUND_1
    settings[const.BOUND_2] = const.DEFAULT_BOUND_2
    settings[const.BOUND_3] = const.DEFAULT_BOUND_3
    settings[const.BOUND_4] = const.DEFAULT_BOUND_4
    settings[const.BOUND_SPIN] = const.DEFAULT_BOUND_SPIN
    settings[const.BOUND_ANGLE] = const.DEFAULT_BOUND_ANGLE
    settings[const.WELL_BORE_STORAGE] = const.DEFAULT_WELL_BORE_STORAGE
    settings[const.INITIAL_WELL_BORE_STORAGE] = const.DEFAULT_INITIAL_WELL_BORE_STORAGE
    settings[const.CHANGING_STORAGE_TIME] = const.DEFAULT_CHANGING_STORAGE_TIME
    settings[const.FRACTURE_HALF_LENGTH] = const.DEFAULT_FRACTURE_HALF_LENGTH
    settings[const.FRACTURE_CONDUCTIVITY] = const.DEFAULT_FRACTURE_CONDUCTIVITY
    settings[const.HORIZONTAL_LENGTH] = const.DEFAULT_HORIZONTAL_LENGTH
    settings[const.HORIZONTAL_CENTER] = const.DEFAULT_HORIZONTAL_CENTER
    settings[const.ANISOTROPY] = const.DEFAULT_ANISOTROPY
    settings[const.FRACTURE_ANGLE] = const.DEFAULT_FRACTURE_ANGLE
    settings[const.FRACTURES_COUNT] = const.DEFAULT_FRACTURES_COUNT
    settings[const.PERFORATING_LENGTH] = const.DEFAULT_PERFORATING_LENGTH
    settings[const.RECTANGLE_WIDTH] = const.DEFAULT_RECTANGLE_WIDTH
    settings[const.RECTANGLE_LENGTH] = const.DEFAULT_RECTANGLE_LENGTH
    settings[const.BOUND_TYPE] = bound
    for key, value in settings_orig.items():
        settings[key] = value
    return settings


@njit
def validate_min_max(param_array, min_max, bound):
    for i in range(0, len(param_array)):
        current = bound.get(param_array[i])
        if current is not None:
            if min_max[i][0] < current[0]:
                min_max[i][0] = current[0]
            if min_max[i][1] > current[1]:
                min_max[i][1] = current[1]


@njit
def create_dynamic_bound_map():
    bound = Dict.empty(key_type=types.int64, value_type=float_array)
    bound[const.FRACTURE_ANGLE] = np.array([const.DOUBLE_MIN_NORMAL, float(const.DEFAULT_ANGEL_MAX)])
    bound[const.FRACTURES_COUNT] = np.array([float(const.DEFAULT_FRACTURES_COUNT_MIN), const.DOUBLE_MAX])
    bound[const.OMEGA] = np.array([const.DOUBLE_MIN_NORMAL, float(const.DEFAULT_RELATIVE_VALUES_MAX)])
    bound[const.LAMBDA] = np.array([const.DOUBLE_MIN_NORMAL, float(const.DEFAULT_RELATIVE_VALUES_MAX)])
    bound[const.BOUND_ANGLE] = np.array([float(const.DEFAULT_ANGEL_MIN), float(const.DEFAULT_ANGEL_MAX)])
    bound[const.WELL_RADIUS] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.PERMEABILITY] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.LAYER_HEIGHT] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.FRACTURE_HALF_LENGTH] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.PERFORATING_LENGTH] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.FRACTURE_CONDUCTIVITY] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.HORIZONTAL_CENTER] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.HORIZONTAL_LENGTH] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.RADIAL_COMPOSITE] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.BOUND_RADIUS] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.BOUND_1] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.BOUND_2] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.BOUND_3] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.BOUND_4] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.RECTANGLE_WIDTH] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.RECTANGLE_LENGTH] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.SKIN] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    bound[const.RADIAL_COMPOSITE_RADIUS] = np.array([float(const.DOUBLE_MIN_NORMAL), float(const.DOUBLE_MAX)])
    return bound


@njit
def get_validated_bounds(params, args, constants):
    rates = args[1]
    settings = args[2]
    param_map = args[3]
    param_map_revers = args[4]
    min_max = args[5]
    dynamic_bound_map = args[6]
    well = constants[0]
    layer = constants[2]
    bound = constants[3]
    is_rate = int(round(constants[4])) == 0
    has_rect_width = param_map_revers.get(const.RECTANGLE_WIDTH) is not None
    has_rect_length = param_map_revers.get(const.RECTANGLE_LENGTH) is not None
    set_params(params, settings, param_map, param_map_revers, rates, is_rate)
    update_by_rect_w_l(settings, has_rect_width, has_rect_length)
    if well == const.VERTICAL or well == const.LIMIT_ENTRY:
        dynamic_bound_map[const.SKIN][0] = const.DEFAULT_SKIN_MIN
        dynamic_bound_map[const.BOUND_RADIUS][0] = settings.get(const.WELL_RADIUS)
        dynamic_bound_map[const.BOUND_1][0] = settings.get(const.WELL_RADIUS)
        dynamic_bound_map[const.BOUND_2][0] = settings.get(const.WELL_RADIUS)
        dynamic_bound_map[const.BOUND_3][0] = settings.get(const.WELL_RADIUS)
        dynamic_bound_map[const.BOUND_4][0] = settings.get(const.WELL_RADIUS)
        dynamic_bound_map[const.RADIAL_COMPOSITE_RADIUS][0] = settings.get(const.WELL_RADIUS)
        dynamic_bound_map[const.HORIZONTAL_LENGTH][0] = settings.get(const.WELL_RADIUS)
        dynamic_bound_map[const.FRACTURE_HALF_LENGTH][0] = settings.get(const.WELL_RADIUS)
        dynamic_bound_map[const.WELL_RADIUS][1] = get_max_by_layer_bound(settings, layer, bound)
        if well == const.LIMIT_ENTRY:
            h = settings.get(const.LAYER_HEIGHT)
            zw = settings.get(const.HORIZONTAL_CENTER)
            hw = settings.get(const.PERFORATING_LENGTH)
            dynamic_bound_map[const.SKIN][0] = 0.0
            dynamic_bound_map[const.HORIZONTAL_CENTER][1] = h - hw / 2.0
            dynamic_bound_map[const.HORIZONTAL_CENTER][0] = hw / 2.0
            dynamic_bound_map[const.PERFORATING_LENGTH][1] = min(zw, h - zw) * 2.0
            dynamic_bound_map[const.LAYER_HEIGHT][0] = zw + hw / 2.0
    elif well == const.UNI_FRACTURE or well == const.INF_FRACTURE or well == const.FIN_FRACTURE:
        dynamic_bound_map[const.SKIN][0] = 0.0
        dynamic_bound_map[const.BOUND_RADIUS][0] = settings.get(const.FRACTURE_HALF_LENGTH)
        dynamic_bound_map[const.BOUND_1][0] = settings.get(const.FRACTURE_HALF_LENGTH)
        dynamic_bound_map[const.BOUND_2][0] = settings.get(const.FRACTURE_HALF_LENGTH)
        dynamic_bound_map[const.BOUND_3][0] = settings.get(const.FRACTURE_HALF_LENGTH)
        dynamic_bound_map[const.BOUND_4][0] = settings.get(const.FRACTURE_HALF_LENGTH)
        dynamic_bound_map[const.RADIAL_COMPOSITE_RADIUS][0] = settings.get(const.FRACTURE_HALF_LENGTH)
        dynamic_bound_map[const.WELL_RADIUS][1] = settings.get(const.FRACTURE_HALF_LENGTH)
        dynamic_bound_map[const.FRACTURE_HALF_LENGTH][1] = get_max_by_layer_bound(settings, layer, bound)
        dynamic_bound_map[const.FRACTURE_CONDUCTIVITY][0] = get_min_fracture_conductivity(settings)
    elif well == const.MULTI_FRAC:
        dynamic_bound_map[const.SKIN][0] = 0.0
        l = settings.get(const.HORIZONTAL_LENGTH)
        xf = settings.get(const.FRACTURE_HALF_LENGTH)
        dynamic_bound_map[const.BOUND_RADIUS][0] = min(l, xf)
        dynamic_bound_map[const.BOUND_1][0] = min(l, xf)
        dynamic_bound_map[const.BOUND_2][0] = min(l, xf)
        dynamic_bound_map[const.BOUND_3][0] = min(l, xf)
        dynamic_bound_map[const.BOUND_4][0] = min(l, xf)
        dynamic_bound_map[const.RADIAL_COMPOSITE_RADIUS][0] = max(l, xf)
        dynamic_bound_map[const.WELL_RADIUS][1] = max(l, xf)
        max_by_layer_bound = get_max_by_layer_bound(settings, layer, bound)
        dynamic_bound_map[const.FRACTURE_HALF_LENGTH][1] = max_by_layer_bound
        if max_by_layer_bound == const.DOUBLE_MAX:
            dynamic_bound_map[const.HORIZONTAL_LENGTH][1] = max_by_layer_bound
        else:
            dynamic_bound_map[const.HORIZONTAL_LENGTH][1] = max_by_layer_bound * 2.0
    elif well == const.HORIZONTAL:
        r = settings.get(const.WELL_RADIUS)
        h = settings.get(const.LAYER_HEIGHT)
        zw = settings.get(const.HORIZONTAL_CENTER)
        dynamic_bound_map[const.SKIN][0] = 0.0
        dynamic_bound_map[const.BOUND_RADIUS][0] = settings.get(const.HORIZONTAL_LENGTH)
        dynamic_bound_map[const.BOUND_1][0] = settings.get(const.HORIZONTAL_LENGTH)
        dynamic_bound_map[const.BOUND_2][0] = settings.get(const.HORIZONTAL_LENGTH)
        dynamic_bound_map[const.BOUND_3][0] = settings.get(const.HORIZONTAL_LENGTH)
        dynamic_bound_map[const.BOUND_4][0] = settings.get(const.HORIZONTAL_LENGTH)
        dynamic_bound_map[const.RADIAL_COMPOSITE_RADIUS][0] = settings.get( const.HORIZONTAL_LENGTH)
        dynamic_bound_map[const.WELL_RADIUS][1] = min(zw, h - zw) * 2.0
        dynamic_bound_map[const.HORIZONTAL_CENTER][0] = r / 2.0
        dynamic_bound_map[const.HORIZONTAL_CENTER][1] = h - r / 2.0
        dynamic_bound_map[const.LAYER_HEIGHT][0] = zw + r / 2.0
        max_by_layer_bound = get_max_by_layer_bound(settings, layer, bound)
        if max_by_layer_bound == const.DOUBLE_MAX:
            dynamic_bound_map[const.HORIZONTAL_LENGTH][1] = max_by_layer_bound
        else:
            dynamic_bound_map[const.HORIZONTAL_LENGTH][1] = max_by_layer_bound * 2.0
    min_max_new = np.copy(min_max)
    for i in range(0, len(params)):
        key = param_map.get(i)
        min_max_dyn = dynamic_bound_map.get(key)
        min_max_cur = min_max_new[i]
        if min_max_dyn is not None:
            min_max_cur[0] = max(min_max_cur[0], min_max_dyn[0])
            min_max_cur[1] = min(min_max_cur[1], min_max_dyn[1])
    return min_max_new


@njit
def update_by_rect_w_l(settings, has_rect_width, has_rect_length):
    if has_rect_width:
        width_half = settings.get(const.RECTANGLE_WIDTH) / 2.0
        settings[const.BOUND_1] = width_half
        settings[const.BOUND_3] = width_half
    if has_rect_length:
        length_half = settings.get(const.RECTANGLE_LENGTH) / 2.0
        settings[const.BOUND_2] = length_half
        settings[const.BOUND_4] = length_half


@njit
def get_max_by_layer_bound(settings, layer, bound):
    if layer == const.RADIAL_COMPOSITE:
        return settings.get(const.RADIAL_COMPOSITE_RADIUS)
    elif bound == const.NO_BOUNDARY:
        return const.DOUBLE_MAX
    elif bound == const.CIRCLE_BOUNDARY_N or bound == const.CIRCLE_BOUNDARY_C:
        return settings.get(const.BOUND_RADIUS)
    elif bound == const.ONE_BOUNDARY_N or bound == const.ONE_BOUNDARY_C:
        return settings.get(const.BOUND_1)
    elif bound == const.CANAL_BOUNDARY_NN or bound == const.CANAL_BOUNDARY_NC or \
            bound == const.CANAL_BOUNDARY_CC or bound == const.INTERSECT_BOUNDARY_NN or \
            bound == const.INTERSECT_BOUNDARY_NC or bound == const.INTERSECT_BOUNDARY_CC:
        d1 = settings.get(const.BOUND_1)
        d2 = settings.get(const.BOUND_2)
        return min(d1, d2)
    elif bound == const.LIMIT_CANAL_BOUNDARY_NNN or bound == const.LIMIT_CANAL_BOUNDARY_NNC or \
            bound == const.LIMIT_CANAL_BOUNDARY_NCN or bound == const.LIMIT_CANAL_BOUNDARY_NNC or \
            bound == const.LIMIT_CANAL_BOUNDARY_CNC or bound == const.LIMIT_CANAL_BOUNDARY_CCC:
        d1 = settings.get(const.BOUND_1)
        d2 = settings.get(const.BOUND_2)
        d3 = settings.get(const.BOUND_3)
        return min(d1, min(d2, d3))
    elif bound == const.RECTANGLE_BOUNDARY_NNNN or bound == const.RECTANGLE_BOUNDARY_NNCC or \
            bound == const.RECTANGLE_BOUNDARY_NCNC or bound == const.RECTANGLE_BOUNDARY_NCCC or \
            bound == const.RECTANGLE_BOUNDARY_NNCN or bound == const.RECTANGLE_BOUNDARY_CCCC:
        d1 = settings.get(const.BOUND_1)
        d2 = settings.get(const.BOUND_2)
        d3 = settings.get(const.BOUND_3)
        d4 = settings.get(const.BOUND_4)
        return min(min(d1, d2), min(d3, d4))
    return const.DOUBLE_MAX


@njit
def get_min_fracture_conductivity(settings):
    xf = settings.get(const.FRACTURE_HALF_LENGTH, const.DEFAULT_FRACTURE_HALF_LENGTH)
    k = settings.get(const.PERMEABILITY, const.DEFAULT_PERMEABILITY)
    return k * xf * const.DEFAULT_FRACTURE_CONDUCTIVITY_MIN


def compile_file():
    cc.compile()
