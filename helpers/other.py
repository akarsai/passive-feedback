#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements a helper class to style print output using
# ansi codes and a helper function to prepare matplotlib figures
# for publication.
#


import matplotlib
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import re

def vmap2d(
        function: callable,
        ) -> callable:
    """
    vectorize a function over two dimensions, so that it accepts a 2d array as input

    Args:
        function: callable to vectorize

    Returns:
        2d vectorized callable
    """

    return jax.vmap(jax.vmap(function, in_axes=0, out_axes=0), in_axes=0, out_axes=0)

def mpl_settings(
        figsize: tuple = (5.5,4),
        backend: str = None,
        latex_font: str = 'computer modern',
        fontsize: int = 10,
        bigger_axis_labels: bool = False,
        dpi: int = 500,
        ) -> None:
    """
    sets matplotlib settings for latex

    :return: None
    """

    plt.rcParams['figure.dpi'] = dpi
    # default for paper: (5.5,4)
    plt.rcParams["figure.figsize"] = figsize
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')

    plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,      # don't setup fonts from rc parameters
            "pgf.preamble": '\\usepackage{amsmath,amssymb}',
            "savefig.transparent": True,
            "font.size": fontsize,
            })

    # make legend font size smaller
    plt.rcParams.update({
        "legend.fontsize": fontsize - 6,
        })

    # bigger axis labels if needed
    if bigger_axis_labels:
        plt.rcParams.update({
            "axes.labelsize": fontsize + 4,
            "axes.titlesize": fontsize + 4,
            })

    if latex_font == 'times':
        plt.rc('font',**{'family':'serif','serif':['Times']})
    elif latex_font == 'computer modern':
        plt.rc('font',**{'family':'serif'})

    plt.rc('axes.formatter', useoffset=False)
    # plt.rcParams['savefig.transparent'] = True

    if backend is not None:
        matplotlib.use(backend)
    if backend == 'macosx':
        plt.rcParams['figure.dpi'] = 140

    return

class style:
    info = '\033[38;5;027m'
    success = '\033[38;5;028m'
    warning = '\033[38;5;208m'
    fail = '\033[38;5;196m'
    #
    bold = '\033[1m'
    underline = '\033[4m'
    italic = '\033[3m'
    end = '\033[0m'



if __name__ == "__main__":
    pass