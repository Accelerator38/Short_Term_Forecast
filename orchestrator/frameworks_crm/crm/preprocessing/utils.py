import numpy as np
import pandas as pd
from itertools import product
from math import isnan
from sklearn.metrics import pairwise_distances
from logging import getLogger

logger = getLogger(__name__)

_DOUBLE_COMPAR_PREC = 1e-5

def train_test_split(data, test_size=0.33):
    train_indx =  int(data.shape[0] * test_size)
    data_train, data_test = data.iloc[:-train_indx], data.iloc[-train_indx-1:]
    return data_train, data_test


def train_test_split_by_date(data, sep_date, format='%d.%m.%y'):
    sep_date = pd.to_datetime(sep_date, format=format)
    assert data.index.contains(sep_date)
    test_size = np.sum(data.index >= sep_date) / len(data)
    return train_test_split(data, test_size)


def pressure_recovery_curve(init_pres, time_points, p_res=200, p_0=10, recov_time=800):
    """
    Computes approximation of the pressure recovery curve after well stop.
    Uses formula: P = P_0 + B * log(t + 1)
    """
    recov_pres = np.zeros(len(time_points))
    recov_pres[0] = init_pres

    B = (p_res - p_0) / np.log(1 + recov_time)
    _t = np.exp((init_pres - p_0) / B) - 1

    for i in range(1, len(time_points)):
        delta = (time_points[i] - time_points[0]).days
        if delta + _t < recov_time:
            recov_pres[i] = p_0 + B * np.log(1 + _t + delta)
        else:
            recov_pres[i] = p_res

    return recov_pres


def recover_pres(pres, inplace=True, p_res=200, p_0=10, recov_time=800):
    """
    Function accepts pressure matrix and fills in zeros according to values from `pressure_recovery_curve`

    Parameters
    ----------
    pres : pandas DataFrame
        Pressure values matrix with time index along rows and wells along columns.
    inplace : bool
        Whether to modify pres or return its modified copy.
    res_pres : float, optional
        Reservoir pressure in Pa.
    recov_time : float, optional
        Recovery time in days, during which pressure almost reaches almost reservoir pressure.

    Returns
    -------
    out : pandas DataFrame
        Matrix of recovered pressure values in Pa.
    """
    logger.info(f"Recovering pressure for wells: {', '.join(pres.columns)}")
    ret_pres = pres if inplace else pres.copy()

    for well_n in ret_pres.columns:
        # don't recover at the beginning and at the end of an array
        cur_pres = np.trim_zeros(ret_pres[well_n])
        if len(cur_pres.index) == 0: continue
        to_recov_indxs = [cur_pres.index[0]]
        for indx in cur_pres.index[1:]:
            if isnan(cur_pres.loc[indx]) or np.abs(cur_pres.loc[indx]) < _DOUBLE_COMPAR_PREC:
                to_recov_indxs.append(indx)
                continue
            elif len(to_recov_indxs) == 1:
                to_recov_indxs = [indx]
                continue
            else:
                init_pres = cur_pres.loc[to_recov_indxs[0]]
                ret_pres.loc[to_recov_indxs, well_n] = pressure_recovery_curve(init_pres, to_recov_indxs,
                                                                               p_res=p_res, p_0=p_0,
                                                                               recov_time=recov_time)
                to_recov_indxs = [indx]
                continue

        if to_recov_indxs:
            init_pres = cur_pres.loc[to_recov_indxs[0]]
            ret_pres.loc[to_recov_indxs, well_n] = pressure_recovery_curve(init_pres, to_recov_indxs,
                                                                           p_res=p_res, p_0=p_0, recov_time=recov_time)

    return ret_pres


def split_by_activity(data, thres=0.4):
    """
    Splits wells by activity criteria.

    Parameters
    ----------
    data : matrix
    thres : float or int
        If integer, then defines number minimum number of time units during which well must work to be considered
        as active.
        If float, then defines percentage of time during which well must work to be considered as active.

    Returns
    -------
    tuple
        tuple of active and inactive wells
    """
    inactivity_time = data[data == 0].count(axis=0)

    if isinstance(thres, float):
        inactivity_time = inactivity_time / len(data.index)

    inactive_wells_n = inactivity_time[inactivity_time >= thres].index
    active_wells_n = inactivity_time.drop(inactive_wells_n).index

    return active_wells_n, inactive_wells_n


def select_by_distance(coordinates, product_n, injection_n, dist_thres=2*1e3):
    """
    Cuts injection wells geographically remote from production ones.

    Parameters
    ----------
    coordinates : pandas DataFrame
    product_n : list or array-like
        Production wells names.
    injection_n : list or array-like
        Injection wells names.
    dist_thres : float, optional
        Maximum distance between infection and production wells at which they are still
        assumed to be interactable.

    Returns
    -------

    """
    pair_dist = pairwise_distances(X=coordinates.loc[product_n],
                                   Y=coordinates.loc[injection_n])
    pair_dist[pair_dist >= dist_thres] = 0.0
    return injection_n[pair_dist.astype(bool).any(axis=0)]


def select_by_distance_double_radius(coordinates, product_n, injection_n, dist_thres=2 * 1e3):
    """
    Cuts injection wells geographically remote from production ones. Consider Top and Bottom of well.

    Parameters
    ----------
    coordinates : pandas DataFrame
    product_n : list or array-like
        Production wells names.
    injection_n : list or array-like
        Injection wells names.
    dist_thres : float, optional
        Maximum distance between infection and production wells at which they are still
        assumed to be interactable.

    Returns
    -------

    """
    def dist(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    pair_dist = np.full((len(product_n), len(injection_n)), dist_thres+1) # исходный массив заполненный нулями
    for prod in range(pair_dist.shape[0]):
        for inj in range(pair_dist.shape[1]):
            prod_x1, prod_x2 = coordinates.loc[product_n[prod]][['Координата X', 'Координата X_1']]
            prod_y1, prod_y2 = coordinates.loc[product_n[prod]][['Координата Y', 'Координата Y_1']]
            inj_x1, inj_x2 = coordinates.loc[injection_n[inj]][['Координата X', 'Координата X_1']]
            inj_y1, inj_y2 = coordinates.loc[injection_n[inj]][['Координата Y', 'Координата Y_1']]

            # угловой коэффициент для добывающего и нагнетательного столба
            try:
                k_prod = (prod_y2 - prod_y1) / (prod_x2 - prod_x1)
            except ZeroDivisionError:
                k_prod = 0
            try:
                k_inj = (inj_y2 - inj_y1) / (inj_x2 - inj_x1)
            except ZeroDivisionError:
                k_inj = 0

            # перебираем координаты по иксу и находим расстояние между двумя точками
            for i in np.linspace(prod_x1, prod_x2, 10):
                for j in np.linspace(inj_x1, inj_x2, 10):
                    res = dist(i, k_prod * (i - prod_x1) + prod_y1,
                               j, k_inj * (j - inj_x1) + inj_y1)
                    if res < pair_dist[prod][inj]:
                        pair_dist[prod][inj] = res
    pair_dist[pair_dist >= dist_thres] = 0

    return injection_n[pair_dist.astype(bool).any(axis=0)]


def reindex_by_day(*args):
    """
    Reindexes arbitrary number of dataframes by day

    Parameters
    ----------
    args

    Returns
    -------

    """
    logger.info(f"Reindexing {len(args)} DataFrames by day")
    ret, first_indx, last_indx = [], [], []
    for arg in args:
        ret.append(arg.resample('D').mean().bfill())
        first_indx.append(arg.index[0])
        last_indx.append(arg.index[-1])

    max_first = np.max(first_indx)
    min_last = np.min(last_indx)
    logger.info(f"Minimum among last indexes: {min_last}, maximum amung first indexes: {max_first}")

    return [r.loc[max_first:min_last] for r in ret]


def cut_inactive_wells(*args, thres=0.4):
    """

    Parameters
    ----------
    args
    thres

    Returns
    -------

    """
    ret = []
    for arg in args:
        active_wells_n, inactive_wells_n = split_by_activity(arg, thres=thres)
        logger.info(f"Inactive wells: {', '.join(inactive_wells_n)}")
        ret.append(arg.loc[:, active_wells_n])
    return ret


def select_wells_nearest_to_prod(injection, production, coordinates, dist_thres=2*1e3, double_R=False):
    """

    Parameters
    ----------
    injection
    production
    coordinates
    dist_thres

    Returns
    -------

    """
    if double_R == False:
        ret = injection[select_by_distance(coordinates, production.columns,
                                             injection.columns, dist_thres=dist_thres)]
        logger.info(f"Number of injection wells after selection by dist: {len(ret.columns)}")
    else:
        ret = injection[select_by_distance_double_radius(coordinates, production.columns,
                                           injection.columns, dist_thres=dist_thres)]
        logger.info(f"Number of injection wells after selection by dist: {len(ret.columns)}")
    return ret


def select_rows_range(start, end, *args):
    """

    Parameters
    ----------
    start
    end
    args

    Returns
    -------

    """
    logger.info(f"Going to select rows from {start} to end {end} for {len(args)} DataFrames")
    ret = []
    for arg in args:
        ret.append(arg.iloc[start:end, :])
    return ret

def filter_by_coordinates_presence(production, injection, pressure, coordinates):
    """

    Parameters
    ----------
    production
    injection
    pressure
    coordinates

    Returns
    -------

    """
    prod_to_drop = [p for p in production.columns if p not in coordinates.index]
    logger.info(f"Going to drop production wells because coordinates are unknown: {prod_to_drop}")
    production = production.drop(prod_to_drop, axis=1)

    inj_to_drop = [i for i in injection.columns if i not in coordinates.index]
    logger.info(f"Going to drop injection wells because coordinates are unknown: {inj_to_drop}")
    injection = injection.drop(inj_to_drop, axis=1)
    pressure = pressure.loc[:, production.columns]

    logger.info(f"Going to drop coordinates which are not in production and injection data")
    coordinates = coordinates.loc[list(set(production.columns).union(set(injection.columns)))]
    return production, injection, pressure, coordinates