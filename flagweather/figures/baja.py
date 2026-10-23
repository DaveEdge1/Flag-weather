"""Figures for the Baja SST analysis."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle

from ..baja import BOX
from ..sources import era5
from .pacific import DIV
from .style import BLUE, GRID, INK, MUTED, ORANGE, SURF, clean

DOM = dict(lat=(5, 40), lon=(220, 262))   # 140W-98W
TYPE_COLOR = {'HU': '#7a2a0c', 'TS': '#eb6834', 'TD': '#f6c3a8'}


def sst_map(b, months, track_month):
    """Mean SST anomaly over `months`, 26 C isotherm for `track_month` (solid) and its 1991-2020 normal (dashed),
    the Baja box, and storm tracks."""
    fig, ax = plt.subplots(figsize=(10, 8.4), facecolor=SURF, layout='constrained')
    a = b.anom.sel(time=months).mean('time').sel(latitude=slice(*DOM['lat']), longitude=slice(*DOM['lon']))
    levels = np.arange(-3, 3.01, 0.5)
    pc = ax.pcolormesh(a.longitude, a.latitude, a, cmap=DIV, norm=BoundaryNorm(levels, DIV.N, extend='both'),
                       shading='auto')
    s = b.sst.sel(time=track_month).sel(latitude=slice(*DOM['lat']), longitude=slice(*DOM['lon']))
    c = b.clim.sel(month=track_month.month).sel(latitude=slice(*DOM['lat']), longitude=slice(*DOM['lon']))
    ax.contour(s.longitude, s.latitude, s, [26], colors=INK, linewidths=2)
    ax.contour(c.longitude, c.latitude, c, [26], colors=INK, linewidths=1.5, linestyles='--')
    ax.add_patch(Rectangle((BOX['lon'][0], BOX['lat'][0]), BOX['lon'][1] - BOX['lon'][0], BOX['lat'][1] - BOX['lat'][0],
                           fill=False, ec=INK, lw=1.2, ls=':'))
    styles = {'Odalys': (0, (4, 2)), 'Polo': '-', 'Rachel': (0, (1, 1.5))}
    for name, g in b.tracks().items():
        x, y = g.lon % 360, g.lat
        ax.plot(x, y, color=INK, lw=1.3, ls=styles.get(name, '-'), zorder=4, label=name)
        for t in ('TD', 'TS', 'HU'):
            m = g.type == t
            ax.scatter(x[m], y[m], s=22, color=TYPE_COLOR[t], edgecolor=INK, lw=0.4, zorder=5)
        last = g.iloc[-1]
        ax.annotate(name, (last.lon % 360, last.lat), xytext=(5, 4), textcoords='offset points', fontsize=9,
                    color=INK, zorder=6)
    try:   # 0.25-degree coastline from the ERA5 land-sea mask cached for event maps
        lsm = era5.land_sea_mask()
        ax.contour(lsm.longitude % 360, lsm.latitude, lsm, [0.5], colors='#3a3936', linewidths=0.8, zorder=3)
    except Exception:
        pass
    ax.legend(title='Track', frameon=False, fontsize=8.5, title_fontsize=8.5, loc='lower left')
    ax.plot(360 - 111.65, 35.2, marker='*', color=INK, ms=12, mec=SURF, zorder=6)
    ax.annotate('Flagstaff', (360 - 111.65, 35.2), xytext=(6, 4), textcoords='offset points', fontsize=9, color=INK)
    ax.set_xlim(*DOM['lon'])
    ax.set_ylim(*DOM['lat'])
    ax.set_aspect(1 / np.cos(np.deg2rad(22)))
    xt = np.arange(220, 263, 10)
    ax.set_xticks(xt, [f'{360 - x}°W' for x in xt])
    ax.set_yticks(np.arange(5, 41, 5), [f'{y}°N' for y in np.arange(5, 41, 5)])
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    cb = fig.colorbar(pc, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label(f'SST anomaly, {months[0]:%b}–{months[-1]:%b %Y} (°C)', color=INK)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    return fig


def box_bars(b, months=(7, 8, 9)):
    """Interactive bars of the Jul-Sep Baja box SST anomaly by year (Plotly); this year in orange."""
    from .interactive import stacked_bars
    s = b.season_series(months)
    s.index = s.index.astype(int)
    colors = [ORANGE if y == b.year else (MUTED if v >= 0 else BLUE) for y, v in s.items()]
    return stacked_bars([dict(series=s, label='Jul–Sep SST anomaly', unit='°C', colors=colors)], panel_height=260)


def history_scatter(b):
    d = b.history()
    total, through, _, _ = b.current_autumn_precip()
    fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor=SURF)
    ax.scatter(d.baja_sep, d.precip, s=30, color=BLUE, alpha=0.8, edgecolor=SURF, lw=0.8, label='1950–' + str(b.year - 1))
    cur = b.box_a[b.box_a.index == f'{b.year}-09-01'].iloc[0]
    ax.scatter([cur], [total], s=90, marker='D', color=ORANGE, edgecolor=INK, lw=0.6, zorder=4,
               label=f'{b.year} (rain through {through:%b} {through.day})')
    k = np.polyfit(d.baja_sep, d.precip, 1)
    xx = np.array([d.baja_sep.min(), d.baja_sep.max()])
    ax.plot(xx, np.polyval(k, xx), color=MUTED, lw=1.5)
    ax.set_xlabel('September SST anomaly, Baja box (°C)')
    ax.set_ylabel('Flagstaff airport Sep–Oct precipitation (in)')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    clean(ax)
    fig.tight_layout()
    return fig
