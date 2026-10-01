import math
import numpy
from numba import njit

import models_transflow_siam.transflow.matrix as matrix
import models_transflow_siam.transflow.constants as const
import models_transflow_siam.transflow.gdi_math as g_math


@njit
def calc(settings, well, wbs, layer, bound, t, is_rate):
    prepare_settings(settings, well)
    td = calc_dl_time(settings, well, t)
    return calc_function(settings, well, wbs, layer, bound, td, is_rate)


@njit
def calc_function(settings, well, wbs, layer, bound, td, is_rate):
    N = const.STEFEST_QUALITY
    sum_n = 0
    for n in range(1, N + 1, 1):
        sum_p = g_math.stephest_left(n)
        t = n * math.log(2) / td
        p = calc_well(settings, well, layer, bound, td, n)
        p = calc_wbs(settings, well, wbs, p, t)
        sum_n = sum_n + sum_p * rate_transform(is_rate, p, t)

    if is_rate:
        return calc_d_rate(settings, sum_n * math.log(2) / td)
    else:
        return calc_d_pressure(settings, sum_n * math.log(2) / td)


@njit
def rate_transform(is_rate, p, t):
    if is_rate:
        return 1.0 / t / t / p
    else:
        return p


@njit
def calc_well(settings, well, layer, bound, t, n):
    ans = 0
    if well == const.VERTICAL:
        ans = calc_vertical(settings, well, layer, t, n)
    elif well == const.HORIZONTAL:
        ans = calc_horizontal(settings, well, layer, t, n)
    elif well == const.INF_FRACTURE:
        ans = calc_frac_inf_uni(settings, well, layer, t, n, const.FRACTURE_INF_XD)
    elif well == const.UNI_FRACTURE:
        ans = calc_frac_inf_uni(settings, well, layer, t, n, const.FRACTURE_UNI_XD)
    elif well == const.FIN_FRACTURE:
        ans = calc_frac_fin(settings, well, layer, t, n)
    elif well == const.LIMIT_ENTRY:
        ans = calc_limit_entry(settings, well, layer, t, n)
    elif well == const.MULTI_FRAC:
        ans = calc_multi_frac(settings, well, layer, t, n)
    ans = ans + calc_bounds(settings, well, bound, layer, t, n)
    return ans


@njit
def calc_vertical(settings, well, layer, t, n):
    t = n * math.log(2) / t
    layer_f = calc_layer(settings, well, layer, t)
    return g_math.bessel_k0(math.sqrt(t * layer_f)) / t / math.sqrt(t * layer_f) \
           / g_math.bessel_k1(math.sqrt(t * layer_f))


@njit
def calc_horizontal(settings, well, layer, t, n):
    k = settings.get(const.PERMEABILITY, const.DEFAULT_PERMEABILITY)
    h = settings.get(const.LAYER_HEIGHT, const.DEFAULT_LAYER_HEIGHT)
    l = settings.get(const.HORIZONTAL_LENGTH, const.DEFAULT_HORIZONTAL_LENGTH)
    zw = settings.get(const.HORIZONTAL_CENTER, const.DEFAULT_HORIZONTAL_CENTER)
    kz = settings.get(const.ANISOTROPY, const.DEFAULT_ANISOTROPY) * k
    r = settings.get(const.WELL_RADIUS, const.DEFAULT_WELL_RADIUS)
    kt = settings.get(const.PERMEABILITY_TOTAL, const.DEFAULT_PERMEABILITY)
    tm = l / r / 2
    xd = 0.732 * math.sqrt(kt / k)
    zwd = zw / h
    zd = zwd + r * (1 + math.sqrt(kz / k)) / 2 / h
    ld = l * math.sqrt(kz / kt) / 2 / h

    layer_f = calc_layer(settings, well, layer, t)
    check1 = tm * math.sqrt(math.log(2) / t * layer_f) * (1 - xd)
    t = n * math.log(2) / t
    layer_f = calc_layer(settings, well, layer, t)
    arg = tm * math.sqrt(t * layer_f)
    if check1 > const.KI1_QUALITY:
        ans = short_time_hor(t, arg, k, kt, xd, tm, layer_f)
    else:
        ans = long_time_hor(t, arg, k, kt, xd)
    ans = ans + horizontal_f1(t, arg, k, kt, zd, zwd, ld) - horizontal_f2(t, k, kt, tm, zd, zwd, xd, ld, layer_f)
    return ans


@njit
def short_time_hor(t, arg, k, kt, xd, tm, layer_f):
    return math.sqrt(k / kt) * (math.pi - g_math.bessel_ki1(arg * (math.sqrt(kt / k) - xd)) - g_math.bessel_ki1(
        arg * (math.sqrt(kt / k) + xd))) / 2 / t / tm / math.sqrt(t * layer_f)


@njit
def long_time_hor(t, arg, k, kt, xd):
    ans = 0
    steps = const.INTEGRAL_QUALITY
    dx = 2 / steps
    for i in range(0, steps):
        x = dx * i + dx / 2 - 1
        ans = ans + g_math.bessel_k0(arg * abs(math.sqrt(kt / k) * x - xd)) * dx
    return ans / 2 / t


@njit
def horizontal_f1(t, arg, k, kt, zd, zwd, ld):
    ans = g_math.bessel_k0((abs(zd - zwd) * arg) / ld) + g_math.bessel_k0((abs(zd + zwd) * arg) / ld)
    iteration = 1
    is_iterate = True
    while is_iterate:
        summand = g_math.bessel_k0((abs(zd - zwd - 2 * iteration) * arg) / ld) + g_math.bessel_k0(
            (abs(zd + zwd - 2 * iteration) * arg) / ld)
        summand = summand + g_math.bessel_k0((abs(zd - zwd + 2 * iteration) * arg) / ld) + g_math.bessel_k0(
            (abs(zd + zwd + 2 * iteration) * arg) / ld)
        ans = ans + summand
        iteration += 1
        is_iterate = summand != 0 and abs(summand / ans) > const.HORIZONTAL_DELTA_QUALITY
    return math.sqrt(k / kt) * (ans / 2 / ld / t - math.pi / 2 / t / arg)


@njit
def horizontal_f2(t, k, kt, tm, zd, zwd, xd, ld, layer_f):
    ans = 0
    iteration = 1
    empty_iter_count = 0
    is_iterate = True
    while is_iterate:
        arg = math.sqrt(tm * tm * t * layer_f + iteration * iteration * math.pi * math.pi * ld * ld)
        summand = math.cos(iteration * math.pi * zd) * math.cos(iteration * math.pi * zwd) \
                  * (g_math.bessel_ki1(arg * (math.sqrt(kt / k) - xd))
                     + g_math.bessel_ki1(arg * (math.sqrt(kt / k) + xd))) / arg
        ans = ans + summand
        iteration += 1
        if summand == 0 or abs(summand / ans) < const.HORIZONTAL_DELTA_QUALITY:
            empty_iter_count += 1
        else:
            empty_iter_count = 0
        is_iterate = empty_iter_count < const.EMPTY_ITERATION_MAX_COUNT
    return ans * math.sqrt(k / kt) / t


@njit
def calc_frac_inf_uni(settings, well, layer, t, n, xd):
    xf = settings.get(const.FRACTURE_HALF_LENGTH, const.DEFAULT_FRACTURE_HALF_LENGTH)
    r = settings.get(const.WELL_RADIUS, const.DEFAULT_WELL_RADIUS)
    tm = xf / r
    layer_f = calc_layer(settings, well, layer, t)
    check = tm * math.sqrt(math.log(2) / t * layer_f) * (1 - xd)
    t = n * math.log(2) / t
    layer_f = calc_layer(settings, well, layer, t)
    return calc_frac_main(t, check > const.KI1_QUALITY, xd, tm, layer_f)


@njit
def calc_frac_fin(settings, well, layer, t, n):
    xf = settings.get(const.FRACTURE_HALF_LENGTH, const.DEFAULT_FRACTURE_HALF_LENGTH)
    r = settings.get(const.WELL_RADIUS, const.DEFAULT_WELL_RADIUS)
    k = settings.get(const.PERMEABILITY, const.DEFAULT_PERMEABILITY)
    fc = settings.get(const.FRACTURE_CONDUCTIVITY, const.DEFAULT_FRACTURE_CONDUCTIVITY)
    fc = fc / k / xf
    tm = xf / r
    xd = xd_fin(fc)
    layer_f = calc_layer(settings, well, layer, t)
    check = tm * math.sqrt(math.log(2) / t * layer_f) * (1 - xd)
    t = n * math.log(2) / t
    layer_f = calc_layer(settings, well, layer, t)
    return calc_frac_main(t, check > const.KI1_QUALITY, xd, tm, layer_f) + calc_plbd(t, fc, tm, layer_f) \
           - calc_plbd(t, math.pi * math.pow(10, 6), tm, layer_f)


@njit
def xd_fin(fc):
    xd = 0.732
    if 0.5 < fc < 10000:
        xd = (0.759919 + 0.465301 * math.log(fc) + 0.562754 * math.pow(math.log(fc), 2) + 0.363093 * math.pow(
            math.log(fc), 3) + 0.0298881 * math.pow(math.log(fc), 4)) / (
                     1 + 0.994770 * math.log(fc) + 0.896679 * math.pow(math.log(fc), 2) + 0.430707 * math.pow(
                 math.log(fc), 3) + 0.0467339 * math.pow(math.log(fc), 4))
    elif fc <= 0.5:
        xd = 0
    return xd


@njit
def calc_plbd(t, fcdi, tm, layer_f):
    psi = calcPsi(t, fcdi, tm, layer_f)
    return math.pi / t / fcdi / psi / math.tanh(psi)


@njit
def calcPsi(t, fcdi, tm, layer_f):
    return math.sqrt(2 * math.sqrt(tm * tm * t * layer_f + tm * math.sqrt(t * layer_f)) / fcdi + t / math.pow(10, 10))


@njit
def calc_frac_main(t, check, xd, tm, layer_f):
    arg = tm * math.sqrt(t * layer_f)
    if check:
        ans = calc_short_time_frac(t, arg, xd, tm, layer_f)
    else:
        ans = calc_long_time_frac(t, arg, xd)
    return ans


@njit
def calc_short_time_frac(t, arg, xd, tm, layer_f):
    return (math.pi - g_math.bessel_ki1(arg * (1 - xd))
            - g_math.bessel_ki1(arg * (1 + xd))) / 2 / t / tm / math.sqrt(t * layer_f)


@njit
def calc_long_time_frac(t, arg, xd):
    ans = 0
    steps = const.INTEGRAL_QUALITY
    dx = 2 / steps
    for i in range(0, steps):
        x = dx * i + dx / 2 - 1
        ans = ans + g_math.bessel_k0(arg * abs(x - xd)) * dx
    return ans / 2 / t


@njit
def calc_limit_entry(settings, well, layer, t, n):
    k = settings.get(const.PERMEABILITY, const.DEFAULT_PERMEABILITY)
    h = settings.get(const.LAYER_HEIGHT, const.DEFAULT_LAYER_HEIGHT)
    hw = settings.get(const.PERFORATING_LENGTH, const.DEFAULT_PERFORATING_LENGTH)
    zw = settings.get(const.HORIZONTAL_CENTER, const.DEFAULT_HORIZONTAL_CENTER)
    kz = settings.get(const.ANISOTROPY, const.DEFAULT_ANISOTROPY) * k
    r = settings.get(const.WELL_RADIUS, const.DEFAULT_WELL_RADIUS)
    hd = h * math.sqrt(k / kz) / r
    hwd = hw * math.sqrt(k / kz) / r
    zwd = zw * math.sqrt(k / kz) / r
    zd = zwd + math.pow(10, 1.000093692 * math.log10(hwd) - 0.42874776)
    t = n * math.log(2) / t
    layer_f = calc_layer(settings, well, layer, t)

    return calc_limit_func(t, hd, hwd, zd, zwd, layer_f)


@njit
def calc_limit_func(t, hd, hwd, zd, zwd, layer_f):
    arg = math.sqrt(t * layer_f)

    ans = 0
    iteration = 1
    is_iterate = True
    while is_iterate:
        summand = g_math.bessel_k0(calc_tn(arg, iteration, hd)) * math.cos(iteration * math.pi * zd / hd) * math.cos(
            iteration * math.pi * zwd / hd) * math.sin(iteration * math.pi * hwd / 2 / hd) / iteration
        ans = ans + summand
        iteration += 1
        is_iterate = iteration < const.LIMIT_ENTRY_MAX_ITERATION
    ans = 4 * ans * hd / math.pi / t / hwd + calc_k10(arg) / t
    return ans


@njit
def calc_tn(arg, n, hd):
    return math.sqrt(arg * arg + math.pi * math.pi * n * n / hd / hd)


@njit
def calc_k10(arg):
    return g_math.bessel_k0(arg) / g_math.bessel_k1(arg) / arg


@njit
def calc_multi_frac(settings, well, layer, t, n):
    r = settings.get(const.WELL_RADIUS, const.DEFAULT_WELL_RADIUS)
    alpha = settings.get(const.FRACTURE_ANGLE, const.DEFAULT_FRACTURE_ANGLE)
    nf = int(settings.get(const.FRACTURES_COUNT, const.DEFAULT_FRACTURES_COUNT))
    xf = settings.get(const.FRACTURE_HALF_LENGTH, const.DEFAULT_FRACTURE_HALF_LENGTH)
    l = settings.get(const.HORIZONTAL_LENGTH, const.DEFAULT_HORIZONTAL_LENGTH)
    tm = xf / r
    xd_fracs = [0.0] * nf
    for i in range(0, nf, 1):
        xd_fracs[i] = l / (nf - 1) / xf * i
    yd_fracs = [0.0] * nf
    alpha_fracs = [0.0] * nf
    for i in range(0, nf, 1):
        alpha_fracs[i] = alpha
    xf_fracs = [0.0] * nf
    for i in range(0, nf, 1):
        xf_fracs[i] = xf
    t = n * math.log(2) / t
    layer_f = calc_layer(settings, well, layer, t)
    arg = tm * math.sqrt(t * layer_f)
    return calc_multifrac_func(arg, t, nf, xd_fracs, yd_fracs, alpha_fracs)


@njit
def calc_multifrac_func(arg, t, nf, xd_fracs, yd_fracs, alpha_fracs):
    a = numpy.zeros((nf + 1, nf + 1))
    b = [0.0] * (nf + 1)
    b[len(b) - 1] = 1.0 / 2.0 / t
    for i in range(0, len(a) - 1, 1):
        a[i][0] = 1.0
    for i in range(1, len(a[0]), 1):
        a[len(a) - 1][i] = 1.0
    frac_values = []
    for i in range(0, nf, 1):
        frac_values.append(-calc_frac_m(arg, xd_fracs[0], xd_fracs[i], yd_fracs[0], yd_fracs[i], alpha_fracs[0],
                                        alpha_fracs[i]))
    symmetric = make_symmetric(frac_values)
    for i in range(0, nf, 1):
        for j in range(0, nf, 1):
            a[i][j + 1] = symmetric[i][j]
    a_inverse = matrix.calc_inverse(a)
    return matrix.multiply_on_vector(a_inverse, b)[0]


@njit
def make_symmetric(array):
    n = len(array)
    sym = numpy.zeros((n, n))
    for i in range(0, n):
        for j in range(i, n):
            sym[i][j] = array[j - i]
            sym[j][i] = array[j - i]
    return sym


@njit
def calc_frac_m(arg, xd_frac1, xd_frac2, yd_frac1, yd_frac2, alpha1, alpha2):
    a = -1
    b = 0.999
    n = const.INTEGRAL_QUALITY * 2
    h = (b - a) / n
    y0 = g_math.bessel_k0(arg * rd_custom(xd_frac1, xd_frac2, yd_frac1, yd_frac2, alpha1, alpha2, a))
    y1 = g_math.bessel_k0(arg * rd_custom(xd_frac1, xd_frac2, yd_frac1, yd_frac2, alpha1, alpha2, b))
    sum_ans = 0.5 * h * (y0 + y1)
    for k in range(1, n, 1):
        sum_ans = sum_ans + h * g_math.bessel_k0(
            arg * rd_custom(xd_frac1, xd_frac2, yd_frac1, yd_frac2, alpha1, alpha2, a + h * k))
    return sum_ans


@njit
def rd_custom(xd_frac1, xd_frac2, yd_frac1, yd_frac2, alpha1, alpha2, x):
    xd = 0.732
    delta_alpha_rad = math.radians(alpha1 - (alpha1 - alpha2))
    return math.sqrt(math.pow((abs(xd_frac1 - xd_frac2)) - (x - xd) * math.cos(delta_alpha_rad), 2.0) + math.pow(
        -(abs(yd_frac1 - yd_frac2)) - (x - xd) * math.sin(delta_alpha_rad), 2.0))


@njit
def calc_wbs(settings, well, wbs, p, t):
    if wbs == const.CONSTANT_WBS:
        return calc_constant_wbs(settings, well, p, t)
    elif wbs == const.FAIR_WBS:
        return calc_fair_wbs(settings, well, p, t)
    else:
        return calc_no_wbs(settings, p, t)


@njit
def calc_no_wbs(settings, p, t):
    s = settings.get(const.SKIN_CALC, const.DEFAULT_SKIN_CALC)
    return p + s / t


@njit
def calc_constant_wbs(settings, well, p, t):
    s = settings.get(const.SKIN_CALC, const.DEFAULT_SKIN_CALC)
    cs = settings.get(const.WELL_BORE_STORAGE, const.DEFAULT_WELL_BORE_STORAGE)
    cs = calc_dl_wbs(settings, well, cs)
    return (1 / t) * ((s + t * p) / (1 + t * cs * (s + t * p)))


@njit
def calc_fair_wbs(settings, well, p, t):
    s = settings.get(const.SKIN_CALC, const.DEFAULT_SKIN_CALC)
    cs = settings.get(const.WELL_BORE_STORAGE, const.DEFAULT_WELL_BORE_STORAGE)
    cf = settings.get(const.INITIAL_WELL_BORE_STORAGE, const.DEFAULT_INITIAL_WELL_BORE_STORAGE)
    tf = settings.get(const.CHANGING_STORAGE_TIME, const.DEFAULT_CHANGING_STORAGE_TIME)
    cs = calc_dl_wbs(settings, well, cs)
    cf = calc_dl_wbs(settings, well, cf)
    tf = calc_dl_time(settings, well, tf) / 3.342
    cfd = tf * (1 / cf - 1 / cs)
    return (t * p + s) * (1 + t * t * cs * (cfd / t - cfd / (t + 1 / tf))) / (t * (1 + cs * t * (t * p + s)))


@njit
def calc_layer(settings, well, layer, t):
    if layer == const.DUAL_POROSITY_PSS \
            or layer == const.DUAL_POROSITY_SLAB \
            or layer == const.DUAL_POROSITY_SPHERE:
        return calc_dual_porosity(settings, well, layer, t)
    else:
        return calc_homogeneous()


@njit
def calc_homogeneous():
    return 1


@njit
def calc_dual_porosity(settings, well, layer, t):
    l = settings.get(const.LAMBDA, const.DEFAULT_LAMBDA)
    w = settings.get(const.OMEGA, const.DEFAULT_OMEGA)
    l = calc_dl_lambda(settings, well, l)
    if layer == const.DUAL_POROSITY_PSS:
        return (t * w * (1 - w) + l) / (t * (1 - w) + l)
    elif layer == const.DUAL_POROSITY_SLAB:
        return w + math.sqrt(l * (1 - w) / 3 / t) * math.tanh(math.sqrt(3 * t * (1 - w) / l))
    elif layer == const.DUAL_POROSITY_SPHERE:
        return w + l * (math.sqrt(15 * (1 - w) * t / l) / math.tanh(math.sqrt(15 * (1 - w) * t / l)) - 1) / 5 / t
    else:
        return 1.0


@njit
def calc_bounds(settings, well, bound, layer, t, n):
    t = n * math.log(2) / t
    if bound == const.ONE_BOUNDARY_C \
            or bound == const.ONE_BOUNDARY_N:
        return calc_one_fault(settings, well, bound, layer, t)
    elif bound == const.INTERSECT_BOUNDARY_CC \
            or bound == const.INTERSECT_BOUNDARY_NC \
            or bound == const.INTERSECT_BOUNDARY_NN \
            or bound == const.CANAL_BOUNDARY_NC \
            or bound == const.CANAL_BOUNDARY_CC:
        return calc_intersecting_bounds(settings, well, bound, layer, t)
    elif bound == const.CANAL_BOUNDARY_NN:
        return calc_chanel_bounds(settings, well, layer, t)
    elif bound == const.CIRCLE_BOUNDARY_C \
            or bound == const.CIRCLE_BOUNDARY_N:
        return calc_circle_bound(settings, well, bound, layer, t)
    elif bound == const.LIMIT_CANAL_BOUNDARY_CCC \
            or bound == const.LIMIT_CANAL_BOUNDARY_CNC \
            or bound == const.LIMIT_CANAL_BOUNDARY_NCC \
            or bound == const.LIMIT_CANAL_BOUNDARY_NCN \
            or bound == const.LIMIT_CANAL_BOUNDARY_NNC \
            or bound == const.LIMIT_CANAL_BOUNDARY_NNN \
            or bound == const.RECTANGLE_BOUNDARY_CCCC \
            or bound == const.RECTANGLE_BOUNDARY_NCCC \
            or bound == const.RECTANGLE_BOUNDARY_NCNC \
            or bound == const.RECTANGLE_BOUNDARY_NNCC \
            or bound == const.RECTANGLE_BOUNDARY_NNCN \
            or bound == const.RECTANGLE_BOUNDARY_NNNN:
        return calc_complex_bounds(settings, well, bound, layer, t)
    elif bound == const.RADIAL_COMPOSITE:
        return calc_radial_composite(settings, well, t)
    else:
        return calc_no_bound()


@njit
def calc_no_bound():
    return 0


@njit
def calc_one_fault(settings, well, bound, layer, t):
    length = settings.get(const.BOUND_1, const.DEFAULT_BOUND_1)
    length = calc_dl_length(settings, well, length)
    layer_f = calc_layer(settings, well, layer, t)
    sign = 1 if bound == const.ONE_BOUNDARY_N else -1
    return sign * g_math.bessel_k0(2 * length * math.sqrt(t * layer_f)) / t


@njit
def calc_intersecting_bounds(settings, well, bound, layer, t):
    l1 = settings.get(const.BOUND_1, const.DEFAULT_BOUND_1)
    l2 = settings.get(const.BOUND_2, const.DEFAULT_BOUND_2)
    n = settings.get(const.BOUND_ANGLE, const.DEFAULT_BOUND_ANGLE)

    if bound == const.CANAL_BOUNDARY_NC or bound == const.CANAL_BOUNDARY_CC:
        n = 90

    l1 = calc_dl_length(settings, well, l1)
    l2 = calc_dl_length(settings, well, l2)
    tetta = math.pi / n
    alpha = math.atan(math.sin(tetta) / (l2 / l1 + math.cos(tetta)))
    betta = tetta - alpha
    radius = l1 / math.sin(alpha)
    layer_f = calc_layer(settings, well, layer, t)

    ans = 0
    sign = 1
    switcher = True
    angle = 0
    i = 0
    while i < n * 2 - 1:
        if bound == const.INTERSECT_BOUNDARY_CC or bound == const.CANAL_BOUNDARY_CC:
            if switcher:
                angle = angle + betta
                sign = -1
                switcher = False
            else:
                angle = angle + alpha
                sign = 1
                switcher = True
        elif bound == const.INTERSECT_BOUNDARY_NC or bound == const.CANAL_BOUNDARY_NC:
            if switcher:
                angle = angle + betta
                sign = sign * -1
                switcher = False
            else:
                angle = angle + alpha
                switcher = True
        else:
            if switcher:
                angle = angle + betta
                switcher = False
            else:
                angle = angle + alpha
                switcher = True
        ans = ans + sign * g_math.bessel_k0(2 * radius * math.sin(angle) * math.sqrt(t * layer_f)) / t
        i = i + 1
    return ans


@njit
def calc_chanel_bounds(settings, well, layer, t):
    l1 = settings.get(const.BOUND_1, const.DEFAULT_BOUND_1)
    l2 = settings.get(const.BOUND_2, const.DEFAULT_BOUND_2)
    l1 = calc_dl_length(settings, well, l1)
    l2 = calc_dl_length(settings, well, l2)
    layer_f = calc_layer(settings, well, layer, t)

    i = 0
    part1 = 0
    part2 = 0
    switcher = True
    ans = 0
    while True:
        if switcher:
            part1 = part1 + l2 * 2
            part2 = part2 + l1 * 2
            switcher = False
        else:
            part1 = part1 + l1 * 2
            part2 = part2 + l2 * 2
            switcher = True

        i = i + 1
        temp = g_math.bessel_k0(part1 * math.sqrt(t * layer_f)) \
               / t + g_math.bessel_k0(part2 * math.sqrt(t * layer_f)) / t
        ans = ans + temp
        if temp == 0 or math.fabs(temp / ans) < const.BOUNDS_DELTA_QUALITY:
            break
        if i > const.SUM_QUALITY:
            break
    return ans


@njit
def calc_circle_bound(settings, well, bound, layer, t):
    re = settings.get(const.BOUND_RADIUS, const.DEFAULT_BOUND_RADIUS)
    re = calc_dl_length(settings, well, re)
    layer_f = calc_layer(settings, well, layer, t)
    if re * math.sqrt(t) < const.CIRCLE_BOUND_I0_MAX:
        ans = 0
        n = const.INTEGRAL_QUALITY
        dx = 2 / n
        for i in range(0, n):
            x = dx * i + dx / 2 - 1
            ans = ans + g_math.bessel_i0(math.sqrt(t * layer_f) * math.fabs(x)) * dx

        if bound == const.CIRCLE_BOUNDARY_N:
            return ans * g_math.bessel_k1(re * math.sqrt(t * layer_f)) \
                   / g_math.bessel_i1(re * math.sqrt(t * layer_f)) / 2 / t
        else:
            return -ans * g_math.bessel_k0(re * math.sqrt(t * layer_f)) \
                   / g_math.bessel_i0(re * math.sqrt(t * layer_f)) / 2 / t
    else:
        return 0


@njit
def calc_complex_bounds(settings, well, bound, layer, t):
    if bound == const.LIMIT_CANAL_BOUNDARY_CCC \
            or bound == const.LIMIT_CANAL_BOUNDARY_CNC \
            or bound == const.LIMIT_CANAL_BOUNDARY_NCC \
            or bound == const.LIMIT_CANAL_BOUNDARY_NCN \
            or bound == const.LIMIT_CANAL_BOUNDARY_NNC \
            or bound == const.LIMIT_CANAL_BOUNDARY_NNN:
        return calc_limit_chanel_bounds(settings, well, bound, layer, t)
    elif bound == const.RECTANGLE_BOUNDARY_CCCC \
            or bound == const.RECTANGLE_BOUNDARY_NCCC \
            or bound == const.RECTANGLE_BOUNDARY_NCNC \
            or bound == const.RECTANGLE_BOUNDARY_NNCC \
            or bound == const.RECTANGLE_BOUNDARY_NNCN \
            or bound == const.RECTANGLE_BOUNDARY_NNNN:
        return calc_rectangle_bounds(settings, well, bound, layer, t)
    else:
        return 0


@njit
def init_spin(settings, bound, l1, l2, l3, l4):
    l1o, l2o, l3o, l4o = l1, l2, l3, l4
    spin = settings.get(const.BOUND_SPIN, const.DEFAULT_BOUND_SPIN)
    if bound == const.LIMIT_CANAL_BOUNDARY_CCC \
            or bound == const.LIMIT_CANAL_BOUNDARY_CNC \
            or bound == const.LIMIT_CANAL_BOUNDARY_NCN \
            or bound == const.LIMIT_CANAL_BOUNDARY_NNN:
        l4 = 0
    elif bound == const.LIMIT_CANAL_BOUNDARY_NCC \
            or bound == const.LIMIT_CANAL_BOUNDARY_NNC:
        if spin == 0:
            l1 = l3o
            l3 = l1o
        l4 = 0
    elif bound == const.RECTANGLE_BOUNDARY_NCCC:
        if spin == 1:
            l1 = l2o
            l2 = l3o
            l3 = l4o
            l4 = l1o
        elif spin == 2:
            l1 = l3o
            l2 = l4o
            l3 = l1o
            l4 = l2o
        elif spin == 3:
            l1 = l4o
            l2 = l1o
            l3 = l2o
            l4 = l3o
    elif bound == const.RECTANGLE_BOUNDARY_NCNC:
        if spin == 0:
            l1 = l4o
            l2 = l1o
            l3 = l2o
            l4 = l3o
    elif bound == const.RECTANGLE_BOUNDARY_NNCC \
            or bound == const.RECTANGLE_BOUNDARY_NNCN:
        if spin == 0:
            l1 = l3o
            l2 = l4o
            l3 = l1o
            l4 = l2o
        elif spin == 1:
            l1 = l4o
            l2 = l1o
            l3 = l2o
            l4 = l3o
        elif spin == 3:
            l1 = l2o
            l2 = l3o
            l3 = l4o
            l4 = l1o

    return l1, l2, l3, l4


@njit
def calc_limit_chanel_bounds(settings, well, bound, layer, t):
    l1 = calc_dl_length(settings, well, settings.get(const.BOUND_1, const.DEFAULT_BOUND_1))
    l2 = calc_dl_length(settings, well, settings.get(const.BOUND_2, const.DEFAULT_BOUND_2))
    l3 = calc_dl_length(settings, well, settings.get(const.BOUND_3, const.DEFAULT_BOUND_3))
    l1, l2, l3, l4 = init_spin(settings, bound, l1, l2, l3, 0)
    layer_f = calc_layer(settings, well, layer, t)
    v = l1 + l3
    h = l2

    i = 1
    ans = calc_rect_start_iteration(bound, t, l2, l3, layer_f)
    while True:
        temp = 0
        cells = numpy.array([[-i, 0], [i, 0]])
        for row in cells:
            temp = temp + calc_rect_iteration(bound, t, l2, l3, v, h, layer_f, row)
        ans = ans + temp
        i = i + 1
        if temp == 0 or math.fabs(temp / ans) < const.BOUNDS_DELTA_QUALITY:
            break
        if i > const.SUM_QUALITY:
            break
    return ans


@njit
def calc_rectangle_bounds(settings, well, bound, layer, t):
    l1 = calc_dl_length(settings, well, settings.get(const.BOUND_1, const.DEFAULT_BOUND_1))
    l2 = calc_dl_length(settings, well, settings.get(const.BOUND_2, const.DEFAULT_BOUND_2))
    l3 = calc_dl_length(settings, well, settings.get(const.BOUND_3, const.DEFAULT_BOUND_3))
    l4 = calc_dl_length(settings, well, settings.get(const.BOUND_4, const.DEFAULT_BOUND_4))
    l1, l2, l3, l4 = init_spin(settings, bound, l1, l2, l3, l4)
    layer_f = calc_layer(settings, well, layer, t)
    v = l1 + l3
    h = l2 + l4

    i = 1
    ans = calc_rect_start_iteration(bound, t, l2, l3, layer_f)
    while True:
        temp = 0
        k = 0
        cells = numpy.zeros((i * 8, 2))
        for j in range(-i, i + 1):
            cells[k] = [-i, j]
            k = k + 1
            cells[k] = [i, j]
            k = k + 1
            if -i < j < i:
                cells[k] = [j, -i]
                k = k + 1
                cells[k] = [j, i]
                k = k + 1
        for row in cells:
            temp = temp + calc_rect_iteration(bound, t, l2, l3, v, h, layer_f, row)
        ans = ans + temp
        i = i + 1
        if temp == 0 or (ans != 0 and math.fabs(temp / ans) < const.BOUNDS_DELTA_QUALITY):
            break
        if i > const.SUM_QUALITY:
            break
    return ans


@njit
def calc_rect_start_iteration(bound, t, l2, l3, layer_f):
    if bound == const.LIMIT_CANAL_BOUNDARY_CCC or bound == const.RECTANGLE_BOUNDARY_CCCC:
        return -g_math.bessel_k0(2 * l2 * math.sqrt(t * layer_f)) / t \
               - g_math.bessel_k0(2 * l3 * math.sqrt(t * layer_f)) / t \
               + g_math.bessel_k0(2 * math.sqrt(l2 * l2 + l3 * l3) * math.sqrt(t * layer_f)) / t
    elif bound == const.LIMIT_CANAL_BOUNDARY_CNC or bound == const.RECTANGLE_BOUNDARY_NNCC:
        return g_math.bessel_k0(2 * l2 * math.sqrt(t * layer_f)) / t \
               - g_math.bessel_k0(2 * l3 * math.sqrt(t * layer_f)) / t \
               - g_math.bessel_k0(2 * math.sqrt(l2 * l2 + l3 * l3) * math.sqrt(t * layer_f)) / t
    elif bound == const.LIMIT_CANAL_BOUNDARY_NCC or bound == const.RECTANGLE_BOUNDARY_NCCC:
        return -g_math.bessel_k0(2 * l2 * math.sqrt(t * layer_f)) / t \
               - g_math.bessel_k0(2 * l3 * math.sqrt(t * layer_f)) / t \
               + g_math.bessel_k0(2 * math.sqrt(l2 * l2 + l3 * l3) * math.sqrt(t * layer_f)) / t
    elif bound == const.LIMIT_CANAL_BOUNDARY_NCN or bound == const.RECTANGLE_BOUNDARY_NCNC:
        return -g_math.bessel_k0(2 * l2 * math.sqrt(t * layer_f)) / t \
               + g_math.bessel_k0(2 * l3 * math.sqrt(t * layer_f)) / t \
               - g_math.bessel_k0(2 * math.sqrt(l2 * l2 + l3 * l3) * math.sqrt(t * layer_f)) / t
    elif bound == const.LIMIT_CANAL_BOUNDARY_NNC or bound == const.RECTANGLE_BOUNDARY_NNCN:
        return g_math.bessel_k0(2 * l2 * math.sqrt(t * layer_f)) / t \
               - g_math.bessel_k0(2 * l3 * math.sqrt(t * layer_f)) / t \
               - g_math.bessel_k0(2 * math.sqrt(l2 * l2 + l3 * l3) * math.sqrt(t * layer_f)) / t
    elif bound == const.LIMIT_CANAL_BOUNDARY_NNN or bound == const.RECTANGLE_BOUNDARY_NNNN:
        ans = g_math.bessel_k0(2 * l2 * math.sqrt(t * layer_f)) / t \
              + g_math.bessel_k0(2 * l3 * math.sqrt(t * layer_f)) / t \
              + g_math.bessel_k0(2 * math.sqrt(l2 * l2 + l3 * l3) * math.sqrt(t * layer_f)) / t
        return ans
    else:
        return 0.0


@njit
def calc_rect_iteration(bound, t, l2, l3, v, h, f, c):
    if bound == const.LIMIT_CANAL_BOUNDARY_CCC or bound == const.RECTANGLE_BOUNDARY_CCCC:
        return g_math.bessel_k0(2 * math.sqrt(c[0] ** 2 * v ** 2 + c[1] ** 2 * h ** 2) * math.sqrt(t * f)) / t \
               - g_math.bessel_k0(2 * math.sqrt((c[0] * v + l3) ** 2 + c[1] ** 2 * h ** 2) * math.sqrt(t * f)) / t \
               - g_math.bessel_k0(2 * math.sqrt(c[0] ** 2 * v ** 2 + (c[1] * h - l2) ** 2) * math.sqrt(t * f)) / t \
               + g_math.bessel_k0(2 * math.sqrt((c[0] * v + l3) ** 2 + (c[1] * h - l2) ** 2) * math.sqrt(t * f)) / t
    elif bound == const.LIMIT_CANAL_BOUNDARY_CNC or bound == const.RECTANGLE_BOUNDARY_NNCC:
        s = 1 if bound == const.LIMIT_CANAL_BOUNDARY_CNC else math.pow(-1, c[0]) * math.pow(-1, c[1])
        return s * g_math.bessel_k0(2 * math.sqrt(c[0] ** 2 * v ** 2 + c[1] ** 2 * h ** 2) * math.sqrt(t * f)) / t \
               - s * g_math.bessel_k0(2 * math.sqrt((c[0] * v + l3) ** 2 + c[1] ** 2 * h ** 2) * math.sqrt(t * f)) / t \
               + s * g_math.bessel_k0(2 * math.sqrt(c[0] ** 2 * v ** 2 + (c[1] * h - l2) ** 2) * math.sqrt(t * f)) / t \
               - s * g_math.bessel_k0(2 * math.sqrt((c[0] * v + l3) ** 2 + (c[1] * h - l2) ** 2) * math.sqrt(t * f)) / t
    elif bound == const.LIMIT_CANAL_BOUNDARY_NCC or bound == const.RECTANGLE_BOUNDARY_NCCC:
        s = math.pow(-1, c[0])
        return s * g_math.bessel_k0(2 * math.sqrt(c[0] ** 2 * v ** 2 + c[1] ** 2 * h ** 2) * math.sqrt(t * f)) / t \
               - s * g_math.bessel_k0(2 * math.sqrt((c[0] * v + l3) ** 2 + c[1] ** 2 * h ** 2) * math.sqrt(t * f)) / t \
               - s * g_math.bessel_k0(2 * math.sqrt(c[0] ** 2 * v ** 2 + (c[1] * h - l2) ** 2) * math.sqrt(t * f)) / t \
               + s * g_math.bessel_k0(2 * math.sqrt((c[0] * v + l3) ** 2 + (c[1] * h - l2) ** 2) * math.sqrt(t * f)) / t
    elif bound == const.LIMIT_CANAL_BOUNDARY_NCN or bound == const.RECTANGLE_BOUNDARY_NCNC:
        return g_math.bessel_k0(2 * math.sqrt(c[0] ** 2 * v ** 2 + c[1] ** 2 * h ** 2) * math.sqrt(t * f)) / t \
               + g_math.bessel_k0(2 * math.sqrt((c[0] * v + l3) ** 2 + c[1] ** 2 * h ** 2) * math.sqrt(t * f)) / t \
               - g_math.bessel_k0(2 * math.sqrt(c[0] ** 2 * v ** 2 + (c[1] * h - l2) ** 2) * math.sqrt(t * f)) / t \
               - g_math.bessel_k0(2 * math.sqrt((c[0] * v + l3) ** 2 + (c[1] * h - l2) ** 2) * math.sqrt(t * f)) / t
    elif bound == const.LIMIT_CANAL_BOUNDARY_NNC or bound == const.RECTANGLE_BOUNDARY_NNCN:
        s = math.pow(-1, c[0])
        return s * g_math.bessel_k0(2 * math.sqrt(c[0] ** 2 * v ** 2 + c[1] ** 2 * h ** 2) * math.sqrt(t * f)) / t \
               - s * g_math.bessel_k0(2 * math.sqrt((c[0] * v + l3) ** 2 + c[1] ** 2 * h ** 2) * math.sqrt(t * f)) / t \
               + s * g_math.bessel_k0(2 * math.sqrt(c[0] ** 2 * v ** 2 + (c[1] * h - l2) ** 2) * math.sqrt(t * f)) / t \
               - s * g_math.bessel_k0(2 * math.sqrt((c[0] * v + l3) ** 2 + (c[1] * h - l2) ** 2) * math.sqrt(t * f)) / t
    elif bound == const.LIMIT_CANAL_BOUNDARY_NNN or bound == const.RECTANGLE_BOUNDARY_NNNN:
        return g_math.bessel_k0(2 * math.sqrt(c[0] ** 2 * v ** 2 + c[1] ** 2 * h ** 2) * math.sqrt(t * f)) / t \
               + g_math.bessel_k0(2 * math.sqrt((c[0] * v + l3) ** 2 + c[1] ** 2 * h ** 2) * math.sqrt(t * f)) / t \
               + g_math.bessel_k0(2 * math.sqrt(c[0] ** 2 * v ** 2 + (c[1] * h - l2) ** 2) * math.sqrt(t * f)) / t \
               + g_math.bessel_k0(2 * math.sqrt((c[0] * v + l3) ** 2 + (c[1] * h - l2) ** 2) * math.sqrt(t * f)) / t
    else:
        return 0.0


@njit
def calc_radial_composite(settings, well, t):
    d = settings.get(const.DIFFUSIVITY_RATIO, const.DEFAULT_DIFFUSIVITY_RATIO)
    m = settings.get(const.MOBILITY_RATIO, const.DEFAULT_MOBILITY_RATIO)
    ri = settings.get(const.RADIAL_COMPOSITE_RADIUS, const.DEFAULT_RADIAL_COMPOSITE_RADIUS)
    ri = calc_dl_length(settings, well, ri)

    s = t * d
    a11 = t * g_math.bessel_i0(math.sqrt(t)) - math.sqrt(t) * g_math.bessel_i1(math.sqrt(t))
    a12 = t * g_math.bessel_k0(math.sqrt(t)) + math.sqrt(t) * g_math.bessel_k1(math.sqrt(t))
    a22 = -g_math.bessel_k1(ri * math.sqrt(t)) / g_math.bessel_i1(ri * math.sqrt(t))
    a23 = math.sqrt(s / t) * g_math.bessel_k1(ri * math.sqrt(s)) / m / g_math.bessel_i1(ri * math.sqrt(t))
    a32 = g_math.bessel_k0(ri * math.sqrt(t)) / g_math.bessel_i0(ri * math.sqrt(t))
    a33 = -g_math.bessel_k0(ri * math.sqrt(s)) / g_math.bessel_i0(ri * math.sqrt(t))
    deltaA = (a22 * a33 - a23 * a32) / t
    delta = a11 * (a22 * a33 - a23 * a32) + a12 * (a23 - a33)
    ans = deltaA * g_math.bessel_i0(math.sqrt(t)) / delta
    if math.isnan(ans):
        return 0
    else:
        return ans


@njit
def calc_dl_time(settings, well, t):
    k = settings.get(const.PERMEABILITY, const.DEFAULT_PERMEABILITY)
    f = settings.get(const.POROSITY, const.DEFAULT_POROSITY)
    m = settings.get(const.VISCOSITY, const.DEFAULT_VISCOSITY)
    ct = settings.get(const.COMPRESSIBILITY_TOTAL, const.DEFAULT_COMPRESSIBILITY)
    r = settings.get(const.WELL_RADIUS, const.DEFAULT_WELL_RADIUS)
    if well == const.VERTICAL:
        s = settings.get(const.SKIN, const.DEFAULT_SKIN)
        r = r * math.exp(-s)
    if well == const.HORIZONTAL:
        k = settings.get(const.PERMEABILITY_TOTAL, const.DEFAULT_PERMEABILITY)
    return 0.00036 * k * t / f / m / ct / r / r


@njit
def calc_dl_wbs(settings, well, cs):
    h = settings.get(const.LAYER_HEIGHT, const.DEFAULT_LAYER_HEIGHT)
    f = settings.get(const.POROSITY, const.DEFAULT_POROSITY)
    ct = settings.get(const.COMPRESSIBILITY_TOTAL, const.DEFAULT_COMPRESSIBILITY)
    r = settings.get(const.WELL_RADIUS, const.DEFAULT_WELL_RADIUS)
    ktkr = 1
    if well == const.VERTICAL:
        s = settings.get(const.SKIN, const.DEFAULT_SKIN)
        r = r * math.exp(-s)
    if well == const.HORIZONTAL:
        ktkr = settings.get(const.PERMEABILITY_TOTAL, const.DEFAULT_PERMEABILITY) \
               / settings.get(const.PERMEABILITY, const.DEFAULT_PERMEABILITY)
    return 0.159 * cs * ktkr / h / f / ct / r / r


@njit
def calc_dl_lambda(settings, well, length):
    if well == const.VERTICAL:
        s = settings.get(const.SKIN, const.DEFAULT_SKIN)
        return length * math.exp(-2 * s)
    elif well == const.HORIZONTAL:
        anisotropy = settings.get(const.PERMEABILITY_TOTAL, const.DEFAULT_PERMEABILITY) \
                     / settings.get(const.PERMEABILITY, const.DEFAULT_PERMEABILITY)
        return length / anisotropy
    else:
        return length


@njit
def calc_dl_length(settings, well, length):
    r = settings.get(const.WELL_RADIUS, const.DEFAULT_WELL_RADIUS)
    anisotropy = 1
    if well == const.VERTICAL:
        s = settings.get(const.SKIN, const.DEFAULT_SKIN)
        r = r * math.exp(-s)
    elif well == const.HORIZONTAL:
        anisotropy = math.sqrt(settings.get(const.PERMEABILITY_TOTAL, const.DEFAULT_PERMEABILITY)
                               / settings.get(const.PERMEABILITY, const.DEFAULT_PERMEABILITY))

    return length * anisotropy / r


@njit
def calc_d_pressure(settings, pd):
    b = settings.get(const.VOLUME_FACTOR, const.DEFAULT_VOLUME_FACTOR)
    m = settings.get(const.VISCOSITY, const.DEFAULT_VISCOSITY)
    k = settings.get(const.PERMEABILITY, const.DEFAULT_PERMEABILITY)
    h = settings.get(const.LAYER_HEIGHT, const.DEFAULT_LAYER_HEIGHT)
    return 18.41 * b * m * pd / k / h


@njit
def calc_d_rate(settings, pd):
    b = settings.get(const.VOLUME_FACTOR, const.DEFAULT_VOLUME_FACTOR)
    m = settings.get(const.VISCOSITY, const.DEFAULT_VISCOSITY)
    k = settings.get(const.PERMEABILITY, const.DEFAULT_PERMEABILITY)
    h = settings.get(const.LAYER_HEIGHT, const.DEFAULT_LAYER_HEIGHT)
    return k * h * pd / 18.41 / b / m


@njit
def prepare_settings(settings, well):
    if well == const.VERTICAL:
        settings[const.SKIN_CALC] = 0.0
    elif well == const.HORIZONTAL:
        k = settings.get(const.PERMEABILITY, const.DEFAULT_PERMEABILITY)
        kz = settings.get(const.ANISOTROPY, const.DEFAULT_ANISOTROPY) * k
        settings[const.PERMEABILITY_TOTAL] = math.pow(k * k * kz, 1.0 / 3.0)
        settings[const.SKIN_CALC] = settings.get(const.SKIN, const.DEFAULT_SKIN)
    else:
        settings[const.SKIN_CALC] = settings.get(const.SKIN, const.DEFAULT_SKIN)
