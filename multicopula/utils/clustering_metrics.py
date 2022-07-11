from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import pairwise_distances, calinski_harabasz_score, silhouette_score

def mia(data_set: np.ndarray, y_labels: np.ndarray) -> np.float64:  # mean_index_adequacy
    """
    Compute Mean index Adequacy (MIA) [1]

    [1] G. Chicco, R. Napoli, and F. Piglione, ``Comparisons among clustering techniques for electricity customer
        classification,'' IEEE Trans. Power Syst., vol. 21, no. 2, pp. 933-940, May 2006, doi: 10.1109/TPWRS.2006.873122.

    Parameters:
    -----------
        data_set: np.ndarray: (s x n) Matrix with the samples and feature. s: Samples , n: Features/variables
        y_labels: np.ndarray: (s, ) 1-D Vector with labels for each sample fo the data_set

    Returns:
    --------
        np.float64:  MIA Score

    """
    le = LabelEncoder()
    labels = le.fit_transform(y_labels)  # Normalize the label numbers e.g. [0,0,3,3,10,10,10] => [0,0,1,1,2,2,2]
    n_labels = len(le.classes_)
    intra_dists = np.zeros(n_labels)
    for k in range(n_labels):
        cluster_k = data_set[labels == k]
        centroid = cluster_k.mean(axis=0)
        # Between members and its centroid
        # # Version 1
        # intra_dists[k] = np.sqrt(np.average(np.square(pairwise_distances(cluster_k, [centroid]))))

        # Version 2
        intra_dists[k] = np.average(np.square(pairwise_distances(cluster_k, [centroid])))

    # return np.sqrt(np.average(np.square(intra_dists)))  # Some papers has the square other not
    # return np.average(np.square(intra_dists))

    # For version 2
    return np.sqrt(np.average(intra_dists))

def mdi(data_set: np.ndarray, y_labels: np.ndarray) -> np.float64:
    """
    Compute Modified Dunn Index (MDI) [1]

    [1] J. C. Dunn, ``Well-separated clusters and optimal fuzzy partitions,''
         J. Cybern., vol. 4, no. 1, pp. 95-104, 1974, doi: 10.1080/01969727408546059.

    Parameters:
    -----------
        data_set: np.ndarray: (s x n) Matrix with the samples and feature. s: Samples , n: Features/variables
        y_labels: np.ndarray: (s, ) 1-D Vector with labels for each sample fo the data_set

    Returns:
    --------
        np.float64:  MDI Score

    """

    le = LabelEncoder()
    labels = le.fit_transform(y_labels)  # Normalize the label numbers
    n_labels = len(le.classes_)
    # intra_dists = np.zeros(n_labels)
    intra_dists_cluster = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(data_set[0])), dtype=np.float)

    for k in range(n_labels):
        cluster_k = data_set[labels == k]
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        # Between members and its centroid
        # intra_dists[k] = np.sqrt(np.average(pairwise_distances(cluster_k, [centroid])))
        # intra_dists[k] = np.average(pairwise_distances(cluster_k, [centroid]))
        # intra_dists[k] = np.sqrt(np.average(np.square(pairwise_distances(cluster_k, [centroid]))))

        # Between all the members in the group
        pairwise_distances_matrix = pairwise_distances(cluster_k)

        if np.allclose(pairwise_distances_matrix, 0):  # 1 member only in the class
            intra_dists_cluster[k] = 0

        else:  # \hat{d}(D_k)
            intra_dists_cluster[k] = np.sqrt(
                np.average(
                    np.square(
                        pairwise_distances_matrix[np.triu_indices_from(pairwise_distances_matrix,
                                                                       k=1)])))

    centroid_distances = pairwise_distances(centroids)
    centroid_distances[centroid_distances == 0] = np.inf  # To avoid to pick up values where i == j

    return np.max(intra_dists_cluster) / np.min(centroid_distances, axis=1).min()

def cdi(data_set, y_labels):
    """
    Compute Clustering dispersion indicator (CDI) [1]

    CDI =  \hat{d}(C)^{-1}  \sqrt{ K^{-1}  \sum^{K}_{k=1}  \hat{d}^2 (D_k)  }

    \hat{D_k} = \sqrt{1/(2M)  \sum^{M}_{m=1}  d^2(x^(m), X)}   where X is all the points in the same cluster k.

    This is calculated summing the upper triangular matrix of the matrix distances between points in the group.
    The sum of the upper triangular makes that the coefficient 1/(2M) became 1/M, because you are not summing
    repetitive distances e.g. (x_1 - x_2)^2 == (x_2 - x_1)^2.

    [1] G. Chicco, R. Napoli, and F. Piglione, ``Comparisons among clustering techniques for electricity customer
        classification,'' IEEE Trans. Power Syst., vol. 21, no. 2, pp. 933-940, May 2006, doi: 10.1109/TPWRS.2006.873122.


    Parameters:
    -----------
        data_set: np.ndarray: (s x n) Matrix with the samples and feature. s: Samples , n: Features/variables
        y_labels: np.ndarray: (s, ) 1-D Vector with labels for each sample fo the data_set

    Returns:
    --------
        np.float64:  CDI Score

    """

    le = LabelEncoder()
    labels = le.fit_transform(y_labels)  # Normalize the label numbers
    n_labels = len(le.classes_)
    # intra_dists = np.zeros(n_labels)
    intra_dists_cluster = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(data_set[0])), dtype=np.float)

    for k in range(n_labels):
        cluster_k = data_set[labels == k]
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        # Between members and its centroid
        # intra_dists[k] = np.average(pairwise_distances(cluster_k, [centroid]))

        # Between all the members in the group
        pairwise_distances_matrix = pairwise_distances(cluster_k)

        if np.allclose(pairwise_distances_matrix, 0):  # 1 member only in the class
            intra_dists_cluster[k] = 0

        else: # \hat{d}(D_k)
            intra_dists_cluster[k] =np.sqrt(
                                        np.average(
                                            np.square(
                                                pairwise_distances_matrix[np.triu_indices_from(pairwise_distances_matrix,
                                                                                               k=1)])))
    centroid_distances = pairwise_distances(centroids)
    intra_centroid_distances = np.sqrt(np.average(np.square(centroid_distances[np.triu_indices_from(centroid_distances,
                                                                                       k=1)] )))

    return (1 / intra_centroid_distances) * np.sqrt(np.average(np.square(intra_dists_cluster)))