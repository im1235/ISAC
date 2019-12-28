import matplotlib.pyplot as plt
import numpy as np


def plot_hist(name, data, data_labels, save=True):
    """plots multiple histograms on same plot, saves plot"""
    plt.figure(name)
    plt.clf()
    min_data = min([d.min() for d in data])
    max_data = max([d.max() for d in data])
    range_data = max_data - min_data
    bins = np.linspace(min_data - 0.1 * range_data, max_data + 0.1 * range_data, 50)

    for d, l in zip(data, data_labels):
        plt.hist(d, bins=bins, alpha=0.2, label=l)

    plt.grid()
    plt.legend()
    plt.ylabel("count")
    plt.title(name)
    if save:
        plt.savefig("results/%s.png" % name, bbox_inches='tight')
