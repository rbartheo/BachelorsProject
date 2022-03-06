# Import necessary libraries
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as shc
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN, MiniBatchKMeans, SpectralClustering, \
    AffinityPropagation
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from preprocessing import data_pca, normalize_data



def k_means(mySeries, nameOfSeries):
    """
    A function that makes K-means clustering. It also calls kmeans_plot function that plots the results of clustering,
    k_means_distribution that prints the distribution of the clusters and scores that print the clustering efficiency
    scores
    :param mySeries: List of data series
    :type mySeries: list
    :param nameOfSeries: List of data series names
    :type nameOfSeries: list
    :return: k_means_distribution: distribution of which data series belongs to which cluster, scores: scores of
    clustering efficiency
    :rtype: list
    """
    mySeries_transformed = data_pca(mySeries)
    cluster_count = math.ceil(math.sqrt(len(mySeries)))
    som_y = math.ceil(math.sqrt(math.sqrt(len(mySeries))))
    kmeans = KMeans(n_clusters=cluster_count, max_iter=5000)
    labels = kmeans.fit_predict(mySeries_transformed)
    kmeans_plot(cluster_count, labels, som_y, mySeries_transformed, mySeries)
    k_means_distribution = kmeans_cluster_distribution(labels, cluster_count, nameOfSeries)
    scores = k_means_score(kmeans, mySeries_transformed)
    return [k_means_distribution, scores]


def kmeans_plot(cluster_count, labels, som_y, mySeries_transformed, mySeries):
    """
    A function that plots an image of clustering results as linear plot and scatter plot and saves it to file
    :param cluster_count: Number of clusters used in data clustering
    :type cluster_count: int
    :param labels: labels of clusters predicted by K-Means algorithm
    :type labels: numpy.ndarray
    :param som_y: rounded number of square rooted length of mySeries list used in plot
    :type som_y: int
    :param mySeries_transformed: List of dataseries transformed by PCA algorithm
    :type mySeries_transformed: list
    :param mySeries: List of data series
    :type mySeries: list
    """
    mySeries = normalize_data(mySeries)
    plot_count = math.ceil(math.sqrt(cluster_count))
    fig, axs = plt.subplots(plot_count, plot_count, figsize=(25, 25))
    fig.suptitle('Clustering results of K-Means algorithm')
    row_i = 0
    column_j = 0
    for label in set(labels):
        cluster = []
        for i in range(len(labels)):
            if labels[i] == label:
                axs[row_i, column_j].plot(mySeries[i], c="gray", alpha=0.4)
                cluster.append(mySeries[i])
        if len(cluster) > 0:
            axs[row_i, column_j].plot(np.average(np.vstack(cluster), axis=0), c="red")
        axs[row_i, column_j].set_title("Cluster " + str(row_i * som_y + column_j + 1))
        column_j += 1
        if column_j % plot_count == 0:
            row_i += 1
            column_j = 0
    plt.savefig("Plots/Kmeans/kmeans_clusteringplot.png")
    plt.close("all")

    plt.figure(figsize=(25, 10))
    plt.scatter(mySeries_transformed[:, 0], mySeries_transformed[:, 1], c=labels, s=300)
    plt.title("K-Means clustering results")
    plt.xlabel("Number of data points")
    plt.ylabel("Number of data points")
    plt.savefig("Plots/Kmeans/kmeans_scatterplot.png")
    plt.close("all")


def kmeans_cluster_distribution(labels, cluster_count, nameOfSeries):
    """
    A function that plots an image of cluster distribution as bar plot. It also prints a distribution of which
    data belongs to which cluster
    :param labels: labels of clusters predicted by K-Means algorithm
    :type labels: numpy.ndarray
    :param cluster_count: A number of clusters used in clustering
    :type cluster_count: int
    :param nameOfSeries: List of data series names
    :type nameOfSeries: list
    :return: A DataFrame which contains a cluster distribution
    :rtype: DataFrame
    """
    cluster_c = [len(labels[labels == i]) for i in range(cluster_count)]
    cluster_n = ["cluster " + str(i + 1) for i in range(cluster_count)]
    plt.figure(figsize=(15, 5))
    plt.title("Cluster Distribution for K-Means")
    plt.ylabel("Number of data in clusters")
    plt.bar(cluster_n, cluster_c)
    plt.savefig("Plots/Kmeans/kmeans_barplot.png")
    plt.close("all")

    fancy_names_for_labels = [f"Cluster {label + 1}" for label in labels]
    k_means_distribution = pd.DataFrame(zip(nameOfSeries, fancy_names_for_labels),
                                        columns=["Series", "Cluster"]).sort_values(by="Cluster").set_index("Series")
    return k_means_distribution


def k_means_score(kmeans, mySeries_transformed):
    """
    A function that prints a clustering efficiency scores
    :param kmeans: DataFrame of data clustered by K-Means algorithm
    :type kmeans: DataFrame
    :param mySeries_transformed: List of data series transformed by PCA algorithm
    :type mySeries_transformed: list
    :return: silhouette: clustering efficency silhouette score, calinski_hrabasz: clustering efficency calinski_hrabasz
    score, davies_bouldin - clustering efficency davies_bouldin score
    :rtype: list
    """
    labels = kmeans.labels_
    silhouette = silhouette_score(mySeries_transformed, labels, metric='euclidean')
    calinski_hrabasz = calinski_harabasz_score(mySeries_transformed, labels)
    davies_bouldin = davies_bouldin_score(mySeries_transformed, labels)
    return [silhouette, calinski_hrabasz, davies_bouldin]


def hierarchical_plot(mySeries_transformed):
    """
    A function that plots a dendrogram plot of data series to check the cluster count for agglomerative clustering
    :param mySeries_transformed: List of data series transformed by PCA algorithm
    :type mySeries_transformed: list
    """
    plt.figure(figsize=(10, 7))
    plt.title("Dendrograms")
    plt.xlabel("Names of data series")
    plt.ylabel("Values of data series")
    shc.dendrogram(shc.linkage(mySeries_transformed, method='ward'))
    plt.savefig("Plots/Agglomerative/agglomerative_hierarchicalplot.png")
    plt.close("all")


def agglomerative(mySeries, nameOfSeries):
    """
    A function that makes Agglomerative clustering. It also calls agglomerative_plot function that plots the results of
    clustering, hierarchical plot to check how many clusters we have to use,
    agglomerative_distribution that prints the distribution of the clusters and scores that print the
    clustering efficiency scores
    :param mySeries: List of data series
    :type mySeries: list
    :param nameOfSeries: List of data series names
    :type nameOfSeries: list
    :return: agglomerative_distribution: distribution of which data series belongs to which cluster, scores: scores of
    clustering efficiency
    :rtype: list
    """
    mySeries_transformed = data_pca(mySeries)
    cluster_count = math.ceil(math.sqrt(len(mySeries)))
    som_y = math.ceil(math.sqrt(math.sqrt(len(mySeries))))
    agglomerative = AgglomerativeClustering(n_clusters=cluster_count)
    labels = agglomerative.fit_predict(mySeries_transformed)
    hierarchical_plot(mySeries_transformed)
    agglomerative_plot(cluster_count, labels, som_y, mySeries_transformed, mySeries)
    agglomerative_distribution = agglomerative_cluster_distribution(labels, cluster_count, nameOfSeries)
    scores = agglomerative_score(agglomerative, mySeries_transformed)
    return [agglomerative_distribution, scores]


def agglomerative_plot(cluster_count, labels, som_y, mySeries_transformed, mySeries):
    """
    A function that plots an image of clustering results as linear plot and scatter plot and saves it to file
    :param cluster_count: Number of clusters used in data clustering
    :type cluster_count: int
    :param labels: labels of clusters predicted by Agglomerative algorithm
    :type labels: numpy.ndarray
    :param som_y: rounded number of square rooted length of mySeries list used in plot
    :type som_y: int
    :param mySeries_transformed: List of dataseries transformed by PCA algorithm
    :type mySeries_transformed: list
    :param mySeries: List of data series
    :type mySeries: list
    """
    mySeries = normalize_data(mySeries)
    plot_count = math.ceil(math.sqrt(cluster_count))
    fig, axs = plt.subplots(plot_count, plot_count, figsize=(25, 25))
    fig.suptitle('Clustering results of Agglomerative algorithm')
    row_i = 0
    column_j = 0
    for label in set(labels):
        cluster = []
        for i in range(len(labels)):
            if (labels[i] == label):
                axs[row_i, column_j].plot(mySeries[i], c="gray", alpha=0.4)
                cluster.append(mySeries[i])
        if len(cluster) > 0:
            axs[row_i, column_j].plot(np.average(np.vstack(cluster), axis=0), c="red")
        axs[row_i, column_j].set_title("Cluster " + str(row_i * som_y + column_j + 1))
        column_j += 1
        if column_j % plot_count == 0:
            row_i += 1
            column_j = 0
    plt.savefig("Plots/Agglomerative/agglomerative_clusteringplot.png")
    plt.close("all")

    plt.figure(figsize=(25, 10))
    plt.scatter(mySeries_transformed[:, 0], mySeries_transformed[:, 1], c=labels, s=300)
    plt.title("Agglomerative clustering results")
    plt.xlabel("Number of data points")
    plt.ylabel("Number of data points")
    plt.savefig("Plots/Agglomerative/agglomerative_scatterplot.png")
    plt.close("all")


def agglomerative_cluster_distribution(labels, cluster_count, nameOfSeries):
    """
    A function that plots an image of cluster distribution as bar plot. It also prints a distribution of which
    data belongs to which cluster
    :param labels: labels of clusters predicted by Agglomerative algorithm
    :type labels: numpy.ndarray
    :param cluster_count: A number of clusters used in clustering
    :type cluster_count: int
    :param nameOfSeries: List of data series names
    :type nameOfSeries: list
    :return: A DataFrame which contains a cluster distribution
    :rtype: DataFrame
    """
    cluster_c = [len(labels[labels == i]) for i in range(cluster_count)]
    cluster_n = ["Cluster " + str(i + 1) for i in range(cluster_count)]
    plt.figure(figsize=(15, 5))
    plt.title("Cluster Distribution for Agglomerative")
    plt.ylabel("Number of data in clusters")
    plt.bar(cluster_n, cluster_c)
    plt.savefig("Plots/Agglomerative/agglomerative_barplot.png")
    plt.close("all")

    fancy_names_for_labels = [f"Cluster {label + 1}" for label in labels]
    agglomerative_distribution = pd.DataFrame(zip(nameOfSeries, fancy_names_for_labels),
                                              columns=["Series", "Cluster"]).sort_values(by="Cluster").set_index(
        "Series")
    return agglomerative_distribution


def agglomerative_score(agglomerative, mySeries_transformed):
    """
    A function that prints a clustering efficiency scores
    :param agglomerative: DataFrame of data clustered by Agglomerative algorithm
    :type agglomerative: DataFrame
    :param mySeries_transformed: List of data series transformed by PCA algorithm
    :type mySeries_transformed: list
    :return: silhouette: clustering efficency silhouette score, calinski_hrabasz: clustering efficency calinski_hrabasz
    score, davies_bouldin - clustering efficency davies_bouldin score
    :rtype: list
    """
    labels = agglomerative.labels_
    silhouette = silhouette_score(mySeries_transformed, labels, metric='euclidean')
    calinski_hrabasz = calinski_harabasz_score(mySeries_transformed, labels)
    davies_bouldin = davies_bouldin_score(mySeries_transformed, labels)
    return [silhouette, calinski_hrabasz, davies_bouldin]


def birch(mySeries, nameOfSeries):
    """
    A function that makes BIRCH clustering. It also calls birch_plot function that plots the results of
    clustering, birch_distribution that prints the distribution of the clusters and scores that print the
    clustering efficiency scores
    :param mySeries: List of data series
    :type mySeries: list
    :param nameOfSeries: List of data series names
    :type nameOfSeries: list
    :return: birch_distribution: distribution of which data series belongs to which cluster, scores: scores of
    clustering efficiency
    :rtype: list
    """
    mySeries_transformed = data_pca(mySeries)
    som_y = math.ceil(math.sqrt(math.sqrt(len(mySeries))))
    cluster_count = math.ceil(math.sqrt(len(mySeries)))
    birch = Birch(threshold=0.5, branching_factor=60, n_clusters=cluster_count)
    labels = birch.fit_predict(mySeries_transformed)
    birch_plot(cluster_count, labels, som_y, mySeries, mySeries_transformed)
    birch_distribution = birch_cluster_distribution(labels, cluster_count, nameOfSeries)
    scores = birch_score(birch, mySeries_transformed)
    return [birch_distribution, scores]


def birch_plot(cluster_count, labels, som_y, mySeries, mySeries_transformed):
    """
    A function that plots an image of clustering results as linear plot and scatter plot and saves it to file
    :param cluster_count: Number of clusters used in data clustering
    :type cluster_count: int
    :param labels: labels of clusters predicted by BIRCH algorithm
    :type labels: numpy.ndarray
    :param som_y: rounded number of square rooted length of mySeries list used in plot
    :type som_y: int
    :param mySeries_transformed: List of dataseries transformed by PCA algorithm
    :type mySeries_transformed: list
    :param mySeries: List of data series
    :type mySeries: list
    """
    mySeries = normalize_data(mySeries)
    plot_count = math.ceil(math.sqrt(cluster_count))
    fig, axs = plt.subplots(plot_count, plot_count, figsize=(25, 25))
    fig.suptitle('Clustering results of BIRCH algorithm')
    row_i = 0
    column_j = 0
    for label in set(labels):
        cluster = []
        for i in range(len(labels)):
            if (labels[i] == label):
                axs[row_i, column_j].plot(mySeries[i], c="gray", alpha=0.4)
                cluster.append(mySeries[i])
        if len(cluster) > 0:
            axs[row_i, column_j].plot(np.average(np.vstack(cluster), axis=0), c="red")
        axs[row_i, column_j].set_title("Cluster " + str(row_i * som_y + column_j + 1))
        column_j += 1
        if column_j % plot_count == 0:
            row_i += 1
            column_j = 0
    plt.savefig("Plots/Birch/birch_clusteringplot.png")
    plt.close("all")

    plt.figure(figsize=(25, 10))
    plt.scatter(mySeries_transformed[:, 0], mySeries_transformed[:, 1], c=labels, s=300)
    plt.title("BIRCH clustering results")
    plt.xlabel("Number of data points")
    plt.ylabel("Number of data points")
    plt.savefig("Plots/Birch/birch_scatterplot.png")
    plt.close("all")


def birch_cluster_distribution(labels, cluster_count, nameOfSeries):
    """
    A function that plots an image of cluster distribution as bar plot. It also prints a distribution of which
    data belongs to which cluster
    :param labels: labels of clusters predicted by BIRCH algorithm
    :type labels: numpy.ndarray
    :param cluster_count: A number of clusters used in clustering
    :type cluster_count: int
    :param nameOfSeries: List of data series names
    :type nameOfSeries: list
    :return: A DataFrame which contains a cluster distribution
    :rtype: DataFrame
    """
    cluster_c = [len(labels[labels == i]) for i in range(cluster_count)]
    cluster_n = ["Cluster " + str(i + 1) for i in range(cluster_count)]
    plt.figure(figsize=(15, 5))
    plt.title("Cluster Distribution for BIRCH")
    plt.ylabel("Number of data in clusters")
    plt.bar(cluster_n, cluster_c)
    plt.savefig("Plots/Birch/birch_barplot.png")
    plt.close("all")

    fancy_names_for_labels = [f"Cluster {label + 1}" for label in labels]
    birch_distribution = pd.DataFrame(zip(nameOfSeries, fancy_names_for_labels),
                                      columns=["Series", "Cluster"]).sort_values(by="Cluster").set_index("Series")
    return birch_distribution


def birch_score(birch, mySeries_transformed):
    """
    A function that prints a clustering efficiency scores
    :param birch: DataFrame of data clustered by BIRCH algorithm
    :type birch: DataFrame
    :param mySeries_transformed: List of data series transformed by PCA algorithm
    :type mySeries_transformed: list
    :return: silhouette: clustering efficency silhouette score, calinski_hrabasz: clustering efficency calinski_hrabasz
    score, davies_bouldin - clustering efficency davies_bouldin score
    :rtype: list
    """
    labels = birch.labels_
    silhouette = silhouette_score(mySeries_transformed, labels, metric='euclidean')
    calinski_hrabasz = calinski_harabasz_score(mySeries_transformed, labels)
    davies_bouldin = davies_bouldin_score(mySeries_transformed, labels)
    return [silhouette, calinski_hrabasz, davies_bouldin]


def nearest_plot(mySeries):
    """
    A function that plots and saves a linear plot using Nearest neighbors algorithm to check how to set variables of
    DBSCAN algorithm
    :param mySeries: List of data series
    :type mySeries: list
    """
    mySeries_transformed = data_pca(mySeries)
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(mySeries_transformed)
    distances, indices = nbrs.kneighbors(mySeries_transformed)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.figure(figsize=(25, 10))
    plt.title("Plot of variables needed for DBSCAN clustering")
    plt.ylabel("eps value")
    plt.xlabel("Number of series")
    plt.plot(distances)
    plt.savefig("Plots/Dbscan/dbscan_varplot.png")
    plt.close("all")


def dbscan(mySeries, nameOfSeries, eps_value, min_samples_value):
    """
    A function that makes DBSCAN clustering. It also calls dbscan_plot function that plots the results of
    clustering, dbscan_distribution that prints the distribution of the clusters and scores that print the
    clustering efficiency scores
    :param eps_value: value used to set value of eps variable in birch algorithm
    :type eps_value: int
    :param min_samples_value: value used to set value of min_samples in DBSCAN algorithm
    :type min_samples_value: int
    :param mySeries: List of data series
    :type mySeries: list
    :param nameOfSeries: List of data series names
    :type nameOfSeries: list
    :return: dbscan_distribution: distribution of which data series belongs to which cluster, scores: scores of
    clustering efficiency
    :rtype: list
    """
    mySeries_transformed = data_pca(mySeries)
    cluster_count = math.ceil(math.sqrt(len(mySeries)))
    som_y = math.ceil(math.sqrt(math.sqrt(len(mySeries))))
    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
    labels = dbscan.fit_predict(mySeries_transformed)
    nearest_plot(mySeries)
    dbscan_plot(cluster_count, labels, som_y, mySeries, mySeries_transformed)
    dbscan_distribution = dbscan_cluster_distribution(labels, cluster_count, nameOfSeries)
    score = dbscan_score(dbscan, mySeries_transformed)
    return [dbscan_distribution, score]



def dbscan_plot(cluster_count, labels, som_y, mySeries, mySeries_transformed):
    """
    A function that plots an image of clustering results as linear plot and scatter plot and saves it to file
    :param cluster_count: Number of clusters used in data clustering
    :type cluster_count: int
    :param labels: labels of clusters predicted by DBSCAN algorithm
    :type labels: numpy.ndarray
    :param som_y: rounded number of square rooted length of mySeries list used in plot
    :type som_y: int
    :param mySeries_transformed: List of data series transformed by PCA algorithm
    :type mySeries_transformed: list
    :param mySeries: List of data series
    :type mySeries: list
    """
    mySeries = normalize_data(mySeries)
    plot_count = math.ceil(math.sqrt(cluster_count))
    fig, axs = plt.subplots(plot_count, plot_count, figsize=(25, 25))
    fig.suptitle('Clustering results of DBSCAN clustering algorithm')
    row_i = 0
    column_j = 0
    for label in set(labels):
        cluster = []
        for i in range(len(labels)):
            if labels[i] == label:
                axs[row_i, column_j].plot(mySeries[i], c="gray", alpha=0.4)
                cluster.append(mySeries[i])
        if len(cluster) > 0:
            axs[row_i, column_j].plot(np.average(np.vstack(cluster), axis=0), c="red")
        axs[row_i, column_j].set_title("Cluster " + str(row_i * som_y + column_j + 1))
        column_j += 1
        if column_j % plot_count == 0:
            row_i += 1
            column_j = 0
    plt.savefig("Plots/Dbscan/dbscan_clusteringplot.png")
    plt.close("all")

    plt.figure(figsize=(25, 10))
    plt.scatter(mySeries_transformed[:, 0], mySeries_transformed[:, 1], c=labels, s=300)
    plt.title("DBSCAN clustering results")
    plt.xlabel("Number of data points")
    plt.ylabel("Number of data points")
    plt.savefig("Plots/Dbscan/dbscan_scatterplot.png")
    plt.close("all")


def dbscan_cluster_distribution(labels, cluster_count, nameOfSeries):
    """
    A function that plots an image of cluster distribution as bar plot. It also prints a distribution of which
    data belongs to which cluster
    :param labels: labels of clusters predicted by DBSCAN algorithm
    :type labels: numpy.ndarray
    :param cluster_count: A number of clusters used in clustering
    :type cluster_count: int
    :param nameOfSeries: List of data series names
    :type nameOfSeries: list
    :return: A DataFrame which contains a cluster distribution
    :rtype: DataFrame
    """
    cluster_c = [len(labels[labels == i]) for i in range(cluster_count)]
    cluster_n = ["Cluster " + str(i + 1) for i in range(cluster_count)]
    plt.figure(figsize=(15, 5))
    plt.title("Cluster Distribution for DBSCAN")
    plt.ylabel("Number of data in clusters")
    plt.bar(cluster_n, cluster_c)
    plt.savefig("Plots/Dbscan/dbscan_barplot.png")
    plt.close("all")

    fancy_names_for_labels = [f"Cluster {label + 1}" for label in labels]
    dbscan_distribution = pd.DataFrame(zip(nameOfSeries, fancy_names_for_labels),
                                       columns=["Series", "Cluster"]).sort_values(by="Cluster").set_index("Series")
    return dbscan_distribution


def dbscan_score(dbscan, mySeries_transformed):
    """
    A function that prints a clustering efficiency scores
    :param dbscan: DataFrame of data clustered by DBSCAN algorithm
    :type dbscan: DataFrame
    :param mySeries_transformed: List of data series transformed by PCA algorithm
    :type mySeries_transformed: list
    :return: silhouette: clustering efficency silhouette score, calinski_hrabasz: clustering efficency calinski_hrabasz
    score, davies_bouldin - clustering efficency davies_bouldin score
    :rtype: list
    """
    labels = dbscan.labels_
    silhouette = silhouette_score(mySeries_transformed, labels, metric='euclidean')
    calinski_hrabasz = calinski_harabasz_score(mySeries_transformed, labels)
    davies_bouldin = davies_bouldin_score(mySeries_transformed, labels)
    return [silhouette, calinski_hrabasz, davies_bouldin]


def mini_kmeans(mySeries, nameOfSeries):
    """
    A function that makes Mini batch K-Means clustering. It also calls minikmeans_plot function that plots the results
    of clustering, hierarchical plot to check how many clusters we have to use,
    minikmeans_distribution that prints the distribution of the clusters and scores that print the
    clustering efficiency scores
    :param mySeries: List of data series
    :type mySeries: list
    :param nameOfSeries: List of data series names
    :type nameOfSeries: list
    :return: minikmeans_distribution: distribution of which data series belongs to which cluster, scores: scores of
    clustering efficiency
    :rtype: list
    """
    mySeries_transformed = data_pca(mySeries)
    som_y = math.ceil(math.sqrt(math.sqrt(len(mySeries))))
    cluster_count = math.ceil(math.sqrt(len(mySeries_transformed)))
    minikmeans = MiniBatchKMeans(n_clusters=cluster_count)
    labels = minikmeans.fit_predict(mySeries_transformed)
    minikmeans_plot(cluster_count, labels, som_y, mySeries, mySeries_transformed)
    minikmeans_distribution = minikmeans_cluster_distribution(labels, cluster_count, nameOfSeries)
    scores = minikmeans_score(minikmeans, mySeries_transformed)
    return [minikmeans_distribution, scores]


def minikmeans_plot(cluster_count, labels, som_y, mySeries, mySeries_transformed):
    """
    A function that plots an image of clustering results as linear plot and scatter plot and saves it to file
    :param cluster_count: Number of clusters used in data clustering
    :type cluster_count: int
    :param labels: labels of clusters predicted by Mini batch K-Means algorithm
    :type labels: numpy.ndarray
    :param som_y: rounded number of square rooted length of mySeries list used in plot
    :type som_y: int
    :param mySeries_transformed: List of dataseries transformed by PCA algorithm
    :type mySeries_transformed: list
    :param mySeries: List of data series
    :type mySeries: list
    """
    mySeries = normalize_data(mySeries)
    plot_count = math.ceil(math.sqrt(cluster_count))
    fig, axs = plt.subplots(plot_count, plot_count, figsize=(25, 25))
    fig.suptitle('Clustering results of Minibatch KMeans clustering algorithm')
    row_i = 0
    column_j = 0
    for label in set(labels):
        cluster = []
        for i in range(len(labels)):
            if labels[i] == label:
                axs[row_i, column_j].plot(mySeries[i], c="gray", alpha=0.4)
                cluster.append(mySeries[i])
        if len(cluster) > 0:
            axs[row_i, column_j].plot(np.average(np.vstack(cluster), axis=0), c="red")
        axs[row_i, column_j].set_title("Cluster " + str(row_i * som_y + column_j + 1))
        column_j += 1
        if column_j % plot_count == 0:
            row_i += 1
            column_j = 0
    plt.savefig("Plots/Minikmeans/minikmeans_clusteringplot.png")
    plt.close("all")

    plt.figure(figsize=(25, 10))
    plt.scatter(mySeries_transformed[:, 0], mySeries_transformed[:, 1], c=labels, s=300)
    plt.title("Mini Batch K-Means clustering results")
    plt.xlabel("Number of data points")
    plt.ylabel("Number of data points")
    plt.savefig("Plots/Minikmeans/minikmeans_scatterplot.png")
    plt.close("all")


def minikmeans_cluster_distribution(labels, cluster_count, nameOfSeries):
    """
    A function that plots an image of cluster distribution as bar plot. It also prints a distribution of which
    data belongs to which cluster
    :param labels: labels of clusters predicted by Mini batch K-Means algorithm
    :type labels: numpy.ndarray
    :param cluster_count: A number of clusters used in clustering
    :type cluster_count: int
    :param nameOfSeries: List of data series names
    :type nameOfSeries: list
    :return: A DataFrame which contains a cluster distribution
    :rtype: DataFrame
    """
    cluster_c = [len(labels[labels == i]) for i in range(cluster_count)]
    cluster_n = ["Cluster " + str(i + 1) for i in range(cluster_count)]
    plt.figure(figsize=(15, 5))
    plt.title("Cluster Distribution for Mini-Batch KMeans")
    plt.ylabel("Number of data in clusters")
    plt.bar(cluster_n, cluster_c)
    plt.savefig("Plots/Minikmeans/minikmeans_barplot.png")
    plt.close("all")

    fancy_names_for_labels = [f"Cluster {label + 1}" for label in labels]
    minikmeans_distribution = pd.DataFrame(zip(nameOfSeries, fancy_names_for_labels),
                                           columns=["Series", "Cluster"]).sort_values(by="Cluster").set_index("Series")
    return minikmeans_distribution


def minikmeans_score(minikmeans, mySeries_transformed):
    """
    A function that prints a clustering efficiency scores
    :param minikmeans: DataFrame of data clustered by Mini Batch K-Means algorithm
    :type minikmeans: DataFrame
    :param mySeries_transformed: List of data series transformed by PCA algorithm
    :type mySeries_transformed: list
    :return: silhouette: clustering efficency silhouette score, calinski_hrabasz: clustering efficency calinski_hrabasz
    score, davies_bouldin - clustering efficency davies_bouldin score
    :rtype: list
    """
    labels = minikmeans.labels_
    silhouette = silhouette_score(mySeries_transformed, labels, metric='euclidean')
    calinski_hrabasz = calinski_harabasz_score(mySeries_transformed, labels)
    davies_bouldin = davies_bouldin_score(mySeries_transformed, labels)
    return [silhouette, calinski_hrabasz, davies_bouldin]


def spectral(mySeries, nameOfSeries):
    """
    A function that makes Spectral clustering. It also calls spectral_plot function that plots the results of
    clustering, spectral_distribution that prints the distribution of the clusters and scores that print the
    clustering efficiency scores
    :param mySeries: List of data series
    :type mySeries: list
    :param nameOfSeries: List of data series names
    :type nameOfSeries: list
    :return: spectral_distribution: distribution of which data series belongs to which cluster, scores: scores of
    clustering efficiency
    :rtype: list
    """
    mySeries_transformed = data_pca(mySeries)
    som_y = math.ceil(math.sqrt(math.sqrt(len(mySeries))))
    cluster_count = math.ceil(math.sqrt(len(mySeries_transformed)))
    spectral = SpectralClustering(n_clusters=cluster_count, n_init=30)
    labels = spectral.fit_predict(mySeries_transformed)
    spectral_plot(cluster_count, labels, som_y, mySeries, mySeries_transformed)
    spectral_distribution = spectral_cluster_distribution(labels, cluster_count, nameOfSeries)
    scores = spectral_score(spectral, mySeries_transformed)
    return [spectral_distribution, scores]


def spectral_plot(cluster_count, labels, som_y, mySeries, mySeries_transformed):
    """
    A function that plots an image of clustering results as linear plot and scatter plot and saves it to file
    :param cluster_count: Number of clusters used in data clustering
    :type cluster_count: int
    :param labels: labels of clusters predicted by Spectral clustering algorithm
    :type labels: numpy.ndarray
    :param som_y: rounded number of square rooted length of mySeries list used in plot
    :type som_y: int
    :param mySeries_transformed: List of dataseries transformed by PCA algorithm
    :type mySeries_transformed: list
    :param mySeries: List of data series
    :type mySeries: list
    """
    mySeries = normalize_data(mySeries)
    plot_count = math.ceil(math.sqrt(cluster_count))
    fig, axs = plt.subplots(plot_count, plot_count, figsize=(25, 25))
    fig.suptitle('Clustering results of Spectral clustering algorithm')
    row_i = 0
    column_j = 0
    for label in set(labels):
        cluster = []
        for i in range(len(labels)):
            if labels[i] == label:
                axs[row_i, column_j].plot(mySeries[i], c="gray", alpha=0.4)
                cluster.append(mySeries[i])
        if len(cluster) > 0:
            axs[row_i, column_j].plot(np.average(np.vstack(cluster), axis=0), c="red")
        axs[row_i, column_j].set_title("Cluster " + str(row_i * som_y + column_j + 1))
        column_j += 1
        if column_j % plot_count == 0:
            row_i += 1
            column_j = 0
    plt.savefig("Plots/Spectral/spectral_clusteringplot.png")
    plt.close("all")

    plt.figure(figsize=(25, 10))
    plt.scatter(mySeries_transformed[:, 0], mySeries_transformed[:, 1], c=labels, s=300)
    plt.title("Spectral clustering results")
    plt.xlabel("Number of data points")
    plt.ylabel("Number of data points")
    plt.savefig("Plots/Spectral/spectral_scatterplot.png")
    plt.close("all")


def spectral_cluster_distribution(labels, cluster_count, nameOfSeries):
    """
    A function that plots an image of cluster distribution as bar plot. It also prints a distribution of which
    data belongs to which cluster
    :param labels: labels of clusters predicted by Spectral clustering algorithm
    :type labels: numpy.ndarray
    :param cluster_count: A number of clusters used in clustering
    :type cluster_count: int
    :param nameOfSeries: List of data series names
    :type nameOfSeries: list
    :return: A DataFrame which contains a cluster distribution
    :rtype: DataFrame
    """
    cluster_c = [len(labels[labels == i]) for i in range(cluster_count)]
    cluster_n = ["Cluster " + str(i + 1) for i in range(cluster_count)]
    plt.figure(figsize=(15, 5))
    plt.title("Cluster Distribution for Spectral clustering")
    plt.ylabel("Number of data in clusters")
    plt.bar(cluster_n, cluster_c)
    plt.savefig("Plots/Spectral/spectral_barplot.png")
    plt.close("all")

    fancy_names_for_labels = [f"Cluster {label + 1}" for label in labels]
    spectral_distribution = pd.DataFrame(zip(nameOfSeries, fancy_names_for_labels),
                                         columns=["Series", "Cluster"]).sort_values(by="Cluster").set_index("Series")
    return spectral_distribution


def spectral_score(spectral, mySeries_transformed):
    """
    A function that prints a clustering efficiency scores
    :param spectral: DataFrame of data clustered by Spectral clustering algorithm
    :type spectral: DataFrame
    :param mySeries_transformed: List of data series transformed by PCA algorithm
    :type mySeries_transformed: list
    :return: silhouette: clustering efficency silhouette score, calinski_hrabasz: clustering efficency calinski_hrabasz
    score, davies_bouldin - clustering efficency davies_bouldin score
    :rtype: list
    """
    labels = spectral.labels_
    silhouette = silhouette_score(mySeries_transformed, labels, metric='euclidean')
    calinski_hrabasz = calinski_harabasz_score(mySeries_transformed, labels)
    davies_bouldin = davies_bouldin_score(mySeries_transformed, labels)
    return [silhouette, calinski_hrabasz, davies_bouldin]


def affinity(mySeries, nameOfSeries):
    """
    A function that makes Affinity propagation clustering. It also calls affinity_plot function that plots the results
    of clustering, affinity_distribution that prints the distribution of the clusters and scores that print the
    clustering efficiency scores
    :param mySeries: List of data series
    :type mySeries: list
    :param nameOfSeries: List of data series names
    :type nameOfSeries: list
    :return: affinity_distribution: distribution of which data series belongs to which cluster, scores: scores of
    clustering efficiency
    :rtype: list
    """
    mySeries_transformed = data_pca(mySeries)
    som_y = math.ceil(math.sqrt(math.sqrt(len(mySeries))))
    cluster_count = math.ceil(math.sqrt(len(mySeries_transformed)))
    affinity_propagation = AffinityPropagation(damping=0.7, random_state=None)
    labels = affinity_propagation.fit_predict(mySeries_transformed)
    affinity_plot(cluster_count, labels, som_y, mySeries, mySeries_transformed)
    affinity_distribution = affinity_cluster_distribution(labels, cluster_count, nameOfSeries)
    score = affinity_score(affinity_propagation, mySeries_transformed)
    return [affinity_distribution, score]


def affinity_plot(cluster_count, labels, som_y, mySeries, mySeries_transformed):
    """
    A function that plots an image of clustering results as linear plot and scatter plot and saves it to file
    :param cluster_count: Number of clusters used in data clustering
    :type cluster_count: int
    :param labels: labels of clusters predicted by Affinity propagation algorithm
    :type labels: numpy.ndarray
    :param som_y: rounded number of square rooted length of mySeries list used in plot
    :type som_y: int
    :param mySeries_transformed: List of dataseries transformed by PCA algorithm
    :type mySeries_transformed: list
    :param mySeries: List of data series
    :type mySeries: list
    """
    mySeries = normalize_data(mySeries)
    plot_count = math.ceil(math.sqrt(cluster_count))
    fig, axs = plt.subplots(plot_count, plot_count, figsize=(25, 25))
    fig.suptitle('Clustering results of Affinity propagation algorithm')
    row_i = 0
    column_j = 0
    for label in set(labels):
        cluster = []
        for i in range(len(labels)):
            if labels[i] == label:
                axs[row_i, column_j].plot(mySeries[i], c="gray", alpha=0.4)
                cluster.append(mySeries[i])
        if len(cluster) > 0:
            axs[row_i, column_j].plot(np.average(np.vstack(cluster), axis=0), c="red")
        axs[row_i, column_j].set_title("Cluster " + str(row_i * som_y + column_j + 1))
        column_j += 1
        if column_j % plot_count == 0:
            row_i += 1
            column_j = 0
    plt.savefig("Plots/Affinity/affinity_clusteringplot.png")
    plt.close("all")

    plt.figure(figsize=(25, 10))
    plt.scatter(mySeries_transformed[:, 0], mySeries_transformed[:, 1], c=labels, s=300)
    plt.title("Affinity propagation clustering results")
    plt.xlabel("Number of data points")
    plt.ylabel("Number of data points")
    plt.savefig("Plots/Affinity/affinity_scatterplot.png")
    plt.close("all")


def affinity_cluster_distribution(labels, cluster_count, nameOfSeries):
    """
    A function that plots an image of cluster distribution as bar plot. It also prints a distribution of which
    data belongs to which cluster
    :param labels: labels of clusters predicted by Spectral clustering algorithm
    :type labels: numpy.ndarray
    :param cluster_count: A number of clusters used in clustering
    :type cluster_count: int
    :param nameOfSeries: List of data series names
    :type nameOfSeries: list
    :return: A DataFrame which contains a cluster distribution
    :rtype: DataFrame
    """
    cluster_c = [len(labels[labels == i]) for i in range(cluster_count)]
    cluster_n = ["Cluster " + str(i + 1) for i in range(cluster_count)]
    plt.figure(figsize=(15, 5))
    plt.title("Cluster Distribution for Affinity propagation")
    plt.ylabel("Number of data in clusters")
    plt.bar(cluster_n, cluster_c)
    plt.savefig("Plots/Affinity/affinity_barplot.png")
    plt.close("all")

    fancy_names_for_labels = [f"Cluster {label + 1}" for label in labels]
    affinity_distribution = pd.DataFrame(zip(nameOfSeries, fancy_names_for_labels),
                                         columns=["Series", "Cluster"]).sort_values(by="Cluster").set_index("Series")
    return affinity_distribution


def affinity_score(affinity_propagation, mySeries_transformed):
    """
    A function that prints a clustering efficiency scores
    :param affinity_propagation: DataFrame of data clustered by Affinity propagation algorithm
    :type affinity_propagation: DataFrame
    :param mySeries_transformed: List of data series transformed by PCA algorithm
    :type mySeries_transformed: list
    :return: silhouette: clustering efficency silhouette score, calinski_hrabasz: clustering efficency calinski_hrabasz
    score, davies_bouldin - clustering efficency davies_bouldin score
    :rtype: list
    """
    labels = affinity_propagation.labels_
    silhouette = silhouette_score(mySeries_transformed, labels, metric='euclidean')
    calinski_hrabasz = calinski_harabasz_score(mySeries_transformed, labels)
    davies_bouldin = davies_bouldin_score(mySeries_transformed, labels)
    return [silhouette, calinski_hrabasz, davies_bouldin]
