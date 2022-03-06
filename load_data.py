import os
import pandas as pd
from preprocessing import nan_in_series

def load_dataset1():
    """
    A function that loads data sets, converts them into data frames (a pandas variable) and saves them into list. It
    also saves their names to the other list. It also calls the nan_in_series function to check if there are missing
    values.
    :return: Three lists: list of data frames, list of series names, list of series with indexes
    :rtype: list
    """
    # Create lists
    mySeries = []
    nameOfSeries = []
    mySeriesIndex = []
    directory = "data1/"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            colnames = ['timestamp', 'x', 'y', 'z']
            df = pd.read_csv(directory + filename, names=colnames, header=None)
            df2 = df
            mySeriesIndex.append(df2)
            df = df.loc[:, ['timestamp', 'x']]
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            df.index = df.index.astype(str)
            mySeries.append(df)
            nameOfSeries.append(filename[:-4])
            nan_in_series(mySeries)
    return [mySeries, nameOfSeries, mySeriesIndex]


def load_dataset2():
    """
    A function that loads data sets, converts them into data frames (a pandas variable) and saves them into list. It
    also saves their names to the other list. It also calls the nan_in_series function to check if there are missing
    values.
    :return: Three lists: list of data frames, list of series names, list of series with indexes
    :rtype: list
    """
    mySeries = []
    nameOfSeries = []
    mySeriesIndex = []
    directory = "data2/"
    # load all .csv files from package
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            colnames = ['date', 'x', 'y', 'z']
            df = pd.read_csv(directory + filename, names=colnames, header=None)
            df2 = df
            df = df.loc[:, ['date', 'y']]
            mySeriesIndex.append(df2)
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            df.index = df.index.astype(str)
            mySeries.append(df)
            nameOfSeries.append(filename[:-4])
            nan_in_series(mySeries)
    return [mySeries, nameOfSeries, mySeriesIndex]


def load_dataset4():
    """
    A function that loads data sets, converts them into data frames (a pandas variable) and saves them into list. It
    also saves their names to the other list. It also calls the nan_in_series function to check if there are missing
    values.
    :return: Three lists: list of data frames, list of series names, list of series with indexes
    :rtype: list
    """
    mySeries = []
    nameOfSeries = []
    mySeriesIndex = []
    directory = "data4/"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(directory + filename)
            df2 = df
            df = df.loc[:, ['Date', 'Open']]
            mySeriesIndex.append(df2)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            df.index = df.index.astype(str)
            mySeries.append(df)
            nameOfSeries.append(filename[:-4])
            nan_in_series(mySeries)
    return [mySeries, nameOfSeries, mySeriesIndex]


def load_dataset5():
    """
    A function that loads data sets, converts them into data frames (a pandas variable) and saves them into list. It
    also saves their names to the other list. It also calls the nan_in_series function to check if there are missing
    values.
    :return: Three lists: list of data frames, list of series names, list of series with indexes
    :rtype: list
    """
    mySeries = []
    nameOfSeries = []
    mySeriesIndex = []
    directory = "data3/"
    # load all .csv files from package
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(directory + filename)
            df2 = df
            df = df.loc[:, ['Date', 'Confirmed']]
            mySeriesIndex.append(df2)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            df.index = df.index.astype(str)
            mySeries.append(df)
            nameOfSeries.append(filename[:-4])
            nan_in_series(mySeries)
    return [mySeries, nameOfSeries, mySeriesIndex]
