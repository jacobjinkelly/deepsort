"""
For creating visualizations.

Modified from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def show_plot(points, save=False, name="plot"):
    """
    Show a scatter plot of the given points.
    """
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    if save:
        if not os.path.isdir("imgs"):
            os.mkdir("imgs")
        plt.savefig("imgs/" + name + ".jpg")
    plt.show()
