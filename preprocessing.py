from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def series_length(mySeries):
    """
    A function that checks the minimum length of a data series and sets the length of all data frames to the minimum
    :param mySeries: list of data series
    :type mySeries: list
    :return: series_length: minimal length of mySeries list, mySeries: list of data series
    :rtype: list
    """
    minimal = 99999999
    for series in mySeries:
        if len(series) < minimal:
            minimal = len(series)

    min_list = []
    for series in mySeries:
        min_list.append(series[:minimal])

    mySeries = min_list

    return [series_length, mySeries]


def nan_in_series(mySeries):
    """
    A function that checks for a missing values in the data
    :param mySeries: List of data series
    :type mySeries: list
    :return: number of missing data values
    :rtype: int
    """
    series_with_nan = 0
    for series in mySeries:
        if series.isnull().sum().sum() > 0:
            series_with_nan += 1
    return series_with_nan


def normalize_data(mySeries):
    """
    A function that normalizes loaded data by their own values
    :param mySeries: List of data series
    :type mySeries: list
    :return: List of normalized data frames
    :rtype: list
    """

    x, mySeries = series_length(mySeries)
    for i in range(len(mySeries)):
        scaler = MinMaxScaler()
        mySeries[i] = MinMaxScaler().fit_transform(mySeries[i])
        mySeries[i] = mySeries[i].reshape(len(mySeries[i]))
    return mySeries


def data_pca(mySeries):
    """
    A function that uses PCA algorithm to solve the problem with the distances between data points
    :param mySeries: A list of data series
    :type mySeries: list
    :return: list of data series transformed by PCA algorithm
    :rtype: list
    """
    pca = PCA(n_components=2)
    mySeries = normalize_data(mySeries)
    mySeries_transformed = pca.fit_transform(mySeries)
    return mySeries_transformed
