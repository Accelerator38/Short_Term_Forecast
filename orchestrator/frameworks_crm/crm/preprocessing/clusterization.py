import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from logging import getLogger

logger = getLogger(__name__)

def clust_wells(coordinates, *args, n_clusters=8): # кластеризация скважин по координатам и с учётом геологии
    """
    Functions performs wells clusterization based on their coordinates and additional info passed in args.

    Parameters
    ----------
    coordinates pd.DataFrame
        well coordinates
    args tuple with pd.DataFrame
        additional info such as geology
    n_clusters int
        default is 8
    """
    logger.info(f"Performing wells clusterization, n_clusters={n_clusters}")
    features = pd.concat([coordinates] + list(args), axis=1, join_axes=[coordinates.index])
    assert not np.any(features.isna()), 'NaNs are not acceptable'
    clusters = KMeans(n_clusters=n_clusters).fit_predict(features)
    # 0 cluster is reserved for parametrs which are shared between clusters
    clusters += 1
    clusters = pd.DataFrame(np.array([features.index, clusters]).T, columns=['well', 'cluster'])
    logger.info(f"Clusters have following distribution: {str(clusters.groupby(by='cluster').count())}")
    clusters = clusters.set_index('well')
    return clusters
