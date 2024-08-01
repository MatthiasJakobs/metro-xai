import matplotlib
import matplotlib.pyplot as plt

COLORMAP = ['#e41a1c','#377eb8','#984ea3','#ff7f00','#cccc33','#f781bf','#4daf4a', '#a65628']

def default_plot(height_fraction=1.0, width_fraction=1.0, subplots=None, sharex=False, sharey=False):
    width_pt = 468.0 / 2
    width_pt *= width_fraction

    if subplots is None:
        subplots = (1, 1)

    inches_per_pt = 1 / 72.27
    golden_ratio = (5**.5 - 1) / 2

    fig_width_inch = width_pt * inches_per_pt
    fig_height_inch = height_fraction * fig_width_inch * golden_ratio * (subplots[0] / subplots[1])

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        "axes.prop_cycle": f"cycler('color', {COLORMAP})",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Lines
        "lines.linewidth": 1,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6
    }
    plt.rcParams.update(tex_fonts)

    return plt.subplots(subplots[0], subplots[1], figsize=(fig_width_inch, fig_height_inch), sharex=sharex, sharey=sharey)

