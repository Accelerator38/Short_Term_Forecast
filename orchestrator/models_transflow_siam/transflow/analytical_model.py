import math
import numpy as np
from numba import njit
from numba.pycc import CC

import models_transflow_siam.transflow.constants as const
import models_transflow_siam.transflow.gdi_math as g_math
import models_transflow_siam.transflow.model_approx as approx
import models_transflow_siam.transflow.well_calculator as well_c


cc = CC('analytical_model')
cc.output_dir = 'aot'


@njit
@cc.export('calc', 'f8[:, :](DictType(i8, f8), i4, i4, i4, i4, f8[:, :], b1)')
def calc(settings, well, wbs, layer, bound, rates, is_rate=False):
    pi = settings.get(const.PRESSURE_INITIAL, const.DEFAULT_PRESSURE_INITIAL)
    ti = settings.get(const.TIME_INITIAL, const.DEFAULT_TIME_INITIAL)
    approximation = approx.create_approximation(calc_well(settings, well, wbs, layer, bound, rates, is_rate))
    pi = 0.0 if is_rate else pi
    return calc_model(approximation, rates, ti, pi, is_rate)


@njit
@cc.export('calc_points', 'f8[:, :](DictType(i8, f8), f8[:], i4, i4, i4, i4, f8[:, :], b1)')
def calc_points(settings, times, well, wbs, layer, bound, rates, is_rate=False):
    pi = settings.get(const.PRESSURE_INITIAL, const.DEFAULT_PRESSURE_INITIAL)
    ti = settings.get(const.TIME_INITIAL, const.DEFAULT_TIME_INITIAL)
    approximation = approx.create_approximation(calc_well(settings, well, wbs, layer, bound, rates, is_rate))
    pi = 0.0 if is_rate else pi
    return calc_model_point(approximation, rates, ti, pi, times, is_rate)


@njit
def calc_well(settings, well, wbs, layer, bound, rates, is_rate):
    time = g_math.auto_time(const.MODEL_START_TIME, rates[-1][0] + rates[-1][1], const.MODEL_QUALITY)
    data = np.zeros((len(time), 2))
    for i in range(0, len(time)):
        data[i][0] = time[i]
        data[i][1] = well_c.calc(settings, well, wbs, layer, bound, time[i], is_rate)
    return data


@njit
def calc_model(approximation, rates, ti, pi, is_rate):
    pressures = np.array([[ti + 0.0, pi + 0.0]])
    for i in range(1, len(rates)):
        temp_rate = np.array([[rates[0][0], rates[0][1], rates[0][2]]])
        for j in range(1, i + 1):
            temp_rate = np.append(temp_rate, np.array([[rates[j][0], rates[j][1], rates[j][2]]]), axis=0)
        cut_rates = temp_rate if is_rate else cut_rate(temp_rate)
        time = g_math.auto_time(const.MODEL_START_TIME, temp_rate[i][1], const.MODEL_QUALITY)
        pressures = calc_test(approximation, cut_rates, time, ti, pi, pressures)
        pi = pressures[-1][1]
        ti = ti + temp_rate[i][1]
    return pressures


@njit
def calc_model_point(approximation, rates, ti, pi, times, is_rate):
    pressures = np.array([[ti + 0.0, pi + 0.0]])
    rates_indices = []
    for t in times:
        rates_indices.append(find_closest_left_rate(rates, t))
    ti_pi = calc_model_pi(approximation, rates, rates_indices[-1], ti, pi, is_rate)
    for i in range(0, len(times)):
        t = times[i]
        end = rates_indices[i]
        temp_rate = np.array([[rates[0][0], rates[0][1], rates[0][2]]])
        for k in range(1, end + 1):
            temp_rate = np.append(temp_rate, np.array([[rates[k][0], rates[k][1], rates[k][2]]]), axis=0)
        cut_rates = temp_rate if is_rate else cut_rate(temp_rate)
        delta_t = t - ti_pi[end][0]
        if abs(delta_t) < 1E-8:
            pressures = np.append(pressures, np.array([[ti_pi[end][0], ti_pi[end][1]]]), axis=0)
        else:
            pressures = calc_test(approximation, cut_rates, [0.0, delta_t, 0.0], ti_pi[end][0], ti_pi[end][1], pressures)
    return delete_first(pressures)


@njit
def calc_model_pi(approximation, rates, end, ti, pi, is_rate):
    ti = ti + 0.0
    pi = pi + 0.0
    pressures = np.array([[ti, pi]])
    ti_pi = [[ti, pi], [ti, pi]]
    for i in range(1, end + 1):
        temp_rate = np.array([[rates[0][0], rates[0][1], rates[0][2]]])
        for j in range(1, i + 1):
            temp_rate = np.append(temp_rate, np.array([[rates[j][0], rates[j][1], rates[j][2]]]), axis=0)
        cut_rates = temp_rate if is_rate else cut_rate(temp_rate)
        t = temp_rate[i][1]
        pressures = calc_test(approximation, cut_rates, [t - 1.0, t, t + 1.0], ti, pi, pressures)
        pi = pressures[-1][1]
        ti = ti + temp_rate[i][1]
        ti_pi.append([ti, pi])
    return ti_pi


@njit
def calc_test(approximation, rates, time, ti, pi, pressures):
    is_recovery = rates[-1][2] == 0
    p0 = 0 if is_recovery else calc_p0(approximation, rates, 0)
    pi = 0 if math.isnan(pi) else pi
    for i in range(1, len(time) - 1):
        t = time[i]
        p = calc_recovery(approximation, rates, t) if is_recovery else calc_stabilization(approximation, rates, p0, t)
        pa = pi + p
        pressures = np.append(pressures, np.array([[ti + t, pa]]), axis=0)
    return pressures


@njit
def calc_recovery(approximation, rates, t):
    temp1 = 0
    temp2 = 0
    for i in range(1, len(rates) - 1):
        temp1 = temp1 + (rates[i][2] - rates[i - 1][2]) \
                * calc_approximation(approximation, rates[-1][0] - rates[i][0])
    for i in range(1, len(rates)):
        temp2 = temp2 + (rates[i][2] - rates[i - 1][2]) \
                * calc_approximation(approximation, rates[-1][0] - rates[i][0] + t)
    return temp1 - temp2


@njit
def calc_stabilization(approximation, rates, p0, t):
    temp = 0
    for i in range(1, len(rates)):
        temp = temp + (rates[i][2] - rates[i - 1][2]) \
               * calc_approximation(approximation, rates[-1][0] - rates[i][0] + t)
    return -temp - p0


@njit
def calc_p0(approximation, rates, p0):
    for i in range(1, len(rates) - 1):
        p0 = p0 - (rates[i][2] - rates[i - 1][2]) \
             * calc_approximation(approximation, rates[-1][0] - rates[i][0])
    return p0


@njit
def calc_approximation(approximation, t):
    return approx.calc_approximation(approximation, t)


@njit
def cut_rate(rate):
    if len(rate) > 10:
        ans = np.array([[rate[0][0], rate[0][1], rate[0][2]]])
        rate_temp = delete_first(rate)
        buffer = np.zeros((len(rate_temp), 4))
        buffer[0][0] = rate_temp[-1][1]
        buffer[0][1] = 0
        buffer[0][2] = rate_temp[-1][2]
        buffer[0][3] = 0
        for i in range(1, len(rate_temp)):
            buffer[i][0] = rate_temp[len(rate_temp) - i - 1][1]
            buffer[i][1] = buffer[i - 1][1] + buffer[i][0]
            buffer[i][2] = rate_temp[len(rate_temp) - i - 1][2]
            buffer[i][3] = i
        key = 0
        start = 0
        is_iterate = True
        while is_iterate:
            start += 1
            key += 1
            buffer[start][3] = start
            is_iterate = (buffer[start][1] < buffer[0][0] and start < len(rate_temp) - 1) or start < 5
        time = math.exp(math.log(buffer[1][1]) + math.log(10) / float(const.RATE_QUALITY))
        for i in range(start, len(buffer)):
            buffer[i][3] = key
            if buffer[i][1] > time or i == len(buffer) - 1:
                key += 1
                time = math.exp(math.log(buffer[i][1]) + math.log(10) / float(const.RATE_QUALITY))
        summ1 = 0
        summ2 = 0
        key = 1
        for i in range(0, len(rate_temp) - 1):
            summ1 = summ1 + rate_temp[i][1]
            summ2 = summ2 + rate_temp[i][1] * rate_temp[i][2]
            if buffer[len(rate_temp) - 1 - i][3] != buffer[len(rate_temp) - 2 - i][3]:
                duration = summ1
                analysis_start_time = 0
                if key != 0:
                    analysis_start_time = ans[key - 1][0] + ans[key - 1][1]
                rate_value = summ2 / summ1
                ans = np.append(ans, np.array([[analysis_start_time, duration, rate_value]]), axis=0)
                key += 1
                summ1 = 0
                summ2 = 0
        ans = np.append(ans, np.array([[rate_temp[-1][0], rate_temp[-1][1], rate_temp[-1][2]]]), axis=0)
    else:
        ans = rate
    return ans


@njit
def delete_first(rr):
    ans = np.zeros((len(rr) - 1, len(rr[0])))
    for i in range(0, len(rr) - 1):
        ans[i] = rr[i + 1]
    return ans


@njit
def find_closest_left_rate(rates, time):
    if time < rates[0][0]:
        return 0
    if time > rates[len(rates) - 1][0]:
        return len(rates) - 1
    lo = 0
    hi = len(rates) - 1
    while lo <= hi:
        mid = int((hi + lo) / 2)
        if time < rates[mid][0]:
            hi = mid - 1
        elif time > rates[mid][0]:
            lo = mid + 1
        else:
            return mid
    return hi


def compile_file():
    cc.compile()
