import pandas as pd
from time_series_utils import Algorithms

def loading_per_cluster(cluster_solutions, data_kw, n_cluster=4, algorithm=Algorithms.AGG_WARD):
    loadings_clusters = list()
    for cluster_label in range(n_cluster):
        idx = cluster_solutions[n_cluster][algorithm] == cluster_label
        loadings_clusters.append(data_kw.loc[:, idx].sum(axis=1))
    loadings_clusters = pd.concat(loadings_clusters, axis=1)
    loadings_clusters.columns = ['cluster_' + str(ii) for ii in range(n_cluster)]
    loadings_clusters = pd.concat([loadings_clusters,
                                   pd.DataFrame(data_kw.sum(axis=1), columns=['sum_total'])], axis=1)

    return loadings_clusters
