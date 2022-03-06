# import necessary libraries
import math
from matplotlib import pyplot as plt


def plot_series(mySeries, nameOfSeries):
    """
    A function that makes a plot of the loaded data series and saves it to a file.
    :param mySeries: list of data frames of data series
    :type mySeries: list
    :param nameOfSeries: list of data series names
    :type nameOfSeries: list
    """
    plot_value = int(math.sqrt(len(mySeries)) + 0.5)
    fig, axs = plt.subplots(plot_value, plot_value, figsize=(25, 25))
    fig.suptitle('Time series data plots')
    for i in range(plot_value):
        for j in range(plot_value):
            if i * plot_value + j + 1 > len(mySeries):
                continue
            axs[i, j].plot(mySeries[i * plot_value + j].values)
            axs[i, j].set_title(nameOfSeries[i * plot_value + j])
    plt.savefig("Plots/whole_series_plot.png")
    plt.close("all")
