import numpy as np
import models_transflow_siam.transflow.analytical_model as model
import models_transflow_siam.transflow.constants as const
from numba import types
from numba.typed import Dict


KIND_CODE = 'kind_code'
K = 'permeability'
_S = 'skin'
R_RES = 'res_radius'
PI = 'pressure_initial'
XF = 'length_half_fracture'
L_HOR = 'length_hor_well_bore'
_N_FRAC = 'number_fractures'
H = 'thickness'
PHI = 'porosity'
CT = 'compressibility_total'
MU = 'viscosity'
B = 'volume_factor'
_KV_KH = 'anisotropy'
_CFD = 'conductivity_fracture_dim'

PARAMS_DEFAULT = {
    KIND_CODE: 0,
    K: 10,
    _S: 0,
    R_RES: 500,
    PI: 350,
    XF: 100,
    L_HOR: 250,
    _N_FRAC: 5,
    H: 10,
    PHI: 0.2,
    CT: 0.00012,
    MU: 1,
    B: 1,
    _KV_KH: 0.1,
    _CFD: 1
}

_RADIUS_WELL = 0.1
_HWF = 0.5

_WBS = const.NO_WBS
_LAYER = const.HOMOGENEOUS


def compute_transient_flow(
        times: np.ndarray,
        regimes: list[tuple[float, float]],
        is_rate: bool,
        kind_code: int = PARAMS_DEFAULT[KIND_CODE],
        thickness: float = PARAMS_DEFAULT[H],
        porosity: float = PARAMS_DEFAULT[PHI],
        compressibility_total: float = PARAMS_DEFAULT[CT],
        viscosity: float = PARAMS_DEFAULT[MU],
        volume_factor: float = PARAMS_DEFAULT[B],
        permeability: float = PARAMS_DEFAULT[K],
        skin: float = PARAMS_DEFAULT[_S],
        res_radius: float = PARAMS_DEFAULT[R_RES],
        pressure_initial: float = PARAMS_DEFAULT[PI],
        length_hor_well_bore: float = PARAMS_DEFAULT[L_HOR],
        length_half_fracture: float = PARAMS_DEFAULT[XF],
        number_fractures: int = PARAMS_DEFAULT[_N_FRAC],
        anisotropy: float = PARAMS_DEFAULT[_KV_KH],
        conductivity_fracture_dim: float = PARAMS_DEFAULT[_CFD],
) -> np.ndarray:

    settings = Dict.empty(key_type=types.int64, value_type=types.double)
    settings[const.WELL_RADIUS] = _RADIUS_WELL
    settings[const.ANISOTROPY] = anisotropy

    settings[const.LAYER_HEIGHT] = thickness
    settings[const.POROSITY] = porosity
    settings[const.COMPRESSIBILITY_TOTAL] = compressibility_total
    settings[const.VISCOSITY] = viscosity
    settings[const.VOLUME_FACTOR] = volume_factor
    settings[const.PERMEABILITY] = permeability
    settings[const.SKIN] = skin
    settings[const.BOUND_RADIUS] = res_radius
    settings[const.PRESSURE_INITIAL] = pressure_initial
    settings[const.HORIZONTAL_CENTER] = thickness * _HWF
    settings[const.HORIZONTAL_LENGTH] = length_hor_well_bore
    settings[const.FRACTURE_HALF_LENGTH] = length_half_fracture
    settings[const.FRACTURE_CONDUCTIVITY] = conductivity_fracture_dim * permeability * length_half_fracture
    settings[const.FRACTURES_COUNT] = number_fractures

    new_times = np.array(times, dtype=np.double)
    new_kind_code = _get_new_kind_code(kind_code)
    boundary_code = const.CIRCLE_BOUNDARY_C
    new_regimes = _prepare_regimes(regimes, pressure_initial, is_rate)

    result = model.calc_points(
        settings,
        new_times,
        new_kind_code,
        _WBS,
        _LAYER,
        boundary_code,
        new_regimes,
        is_rate,
    )
    return np.array([v[1] for v in result])


def _get_new_kind_code(kind_code: int) -> int:
    if kind_code == 0:
        return const.VERTICAL
    elif kind_code == 1:
        return const.HORIZONTAL
    elif kind_code == 2:
        return const.FIN_FRACTURE
    elif kind_code == 3:
        return const.MULTI_FRAC


def _prepare_regimes(regimes: list[tuple[float, float]], pressure_initial: float, is_rate: bool) -> np.ndarray:
    new_regimes = [[0.0, 0.0, pressure_initial]] if is_rate else [[0.0, 0.0, 0.0]]
    for i in range(0, len(regimes)):
        new_regimes.append([new_regimes[i][0] + new_regimes[i][1], regimes[i][0] * 24, regimes[i][1]])
    return np.array(new_regimes)
