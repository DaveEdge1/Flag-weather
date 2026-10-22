"""Shared figure palette and axis styling."""
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURF = '#0b0b0b', '#898781', '#eeeeec', '#fcfcfb'
BLUE, ORANGE, BLUE_LIGHT = '#2a78d6', '#eb6834', '#cde2fb'
IVT_CMAP = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    'ivt', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])


def clean(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    for s in ax.spines.values():
        s.set_visible(False)
