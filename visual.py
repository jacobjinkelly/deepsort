"""For creating visualizations.
"""
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from evaluate import evaluate


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


def show_attention(input_sentence, output_words, attentions):
    """
    Render a visualization of the attention mechanism on the inputs.
    """
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluate_and_show_attention(input_sentence, encoder, decoder):
    """
    Do inference on an input, and display a visualization of the attention mechanism.
    """
    output_words, attentions = evaluate(
        encoder, decoder, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)
