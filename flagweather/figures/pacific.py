"""Figures for the Pacific page: anomaly maps, analog composites, correlation maps, index time series."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

from ..pacific import DOMAIN
from .style import GRID, INK, MUTED, SURF, clean

DIV = LinearSegmentedColormap.from_list(
    'div', ['#0d366b', '#1c5cab', '#5598e7', '#b7d3f6', '#f0efec', '#f6c3a8', '#eb6834', '#b8451b', '#7a2a0c'])
COOL, WARM = '#2a78d6', '#eb6834'
BLUE_ACC = '#2a78d6'
SEASONS = {1: 'DJF', 2: 'JFM', 3: 'FMA', 4: 'MAM', 5: 'AMJ', 6: 'MJJ', 7: 'JJA', 8: 'JAS', 9: 'ASO', 10: 'SON',
           11: 'OND', 12: 'NDJ'}   # ONI season centred on each month
FIELDS = {
    'sst': dict(levels=np.arange(-3, 3.01, 0.5), unit='°C', label='SST anomaly'),
    'msl': dict(levels=np.arange(-12, 12.01, 2), unit='hPa', label='Sea-level pressure anomaly'),
    'z500': dict(levels=np.arange(-150, 151, 25), unit='m', label='500-hPa height anomaly'),
    'r': dict(levels=np.arange(-0.6, 0.61, 0.1), unit='', label='Correlation with Flagstaff Nov–Mar precipitation'),
}


def _frame(ax, lsm):
    ax.contour(lsm.longitude, lsm.latitude, lsm, [0.5], colors='#52514e', linewidths=0.6)
    ax.set_xlim(*DOMAIN['lon'])
    ax.set_ylim(*DOMAIN['lat'])
    ax.set_aspect(1 / np.cos(np.deg2rad(35)))
    xt = np.arange(120, 301, 30)
    ax.set_xticks(xt, [f'{(x if x <= 180 else 360 - x):.0f}°{"E" if x < 180 else ("W" if x > 180 else "")}' for x in xt])
    ax.set_yticks(np.arange(-10, 71, 20), [f'{abs(y)}°{"N" if y >= 0 else "S"}' for y in np.arange(-10, 71, 20)])
    ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.plot(360 - 111.65, 35.2, marker='*', color=INK, ms=10, mec=SURF, mew=0.8, zorder=6)
    for s in ax.spines.values():
        s.set_color(GRID)


def field_map(ax, da, kind, lsm, title, contour=None, contour_levels=None, centres=None, sig=None):
    f = FIELDS[kind]
    norm = BoundaryNorm(f['levels'], DIV.N, extend='both')
    d = da.sel(latitude=slice(*DOMAIN['lat']), longitude=slice(*DOMAIN['lon']))
    pc = ax.pcolormesh(d.longitude, d.latitude, d, cmap=DIV, norm=norm, shading='auto')
    if contour is not None:
        c = contour.sel(latitude=slice(*DOMAIN['lat']), longitude=slice(*DOMAIN['lon']))
        cs = ax.contour(c.longitude, c.latitude, c, contour_levels, colors=INK, linewidths=0.6)
        ax.clabel(cs, cs.levels[::2], fontsize=6.5, fmt='%d')
    if sig is not None:
        ax.contour(d.longitude, d.latitude, np.abs(d), [sig], colors=INK, linewidths=1.0, linestyles='--')
    for lab, (lat, lon) in (centres or {}).items():
        ax.text(lon, lat, lab, color=INK, fontsize=14, fontweight='bold', ha='center', va='center', zorder=7)
    _frame(ax, lsm)
    ax.set_title(title, loc='left', fontsize=10, color=INK)
    return pc


def _colorbar(fig, pc, axes, kind):
    f = FIELDS[kind]
    cb = fig.colorbar(pc, ax=axes, orientation='vertical', shrink=0.85, aspect=25, pad=0.01)
    cb.set_label(f'{f["label"]}{" (" + f["unit"] + ")" if f["unit"] else ""}', color=INK, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=MUTED)


def _centres(p, months):
    nph = p.nph.loc[months].mean()
    return {'H': (nph.lat, nph.lon)}


def anomaly_maps(p, periods):
    """Rows: SST, sea-level pressure, 500-hPa height; columns: periods (list of (label, months))."""
    fig, axes = plt.subplots(3, len(periods), figsize=(7.2 * len(periods) + 1.2, 13), facecolor=SURF,
                             layout='constrained', squeeze=False)
    for j, (lab, months) in enumerate(periods):
        sst, z, msl_a, msl = p.mean_anom(months)
        z_tot = p.atm.z500.sel(time=months).mean('time')
        pcs = [field_map(axes[0, j], sst, 'sst', p.lsm, f'SST anomaly, {lab}'),
               field_map(axes[1, j], msl_a, 'msl', p.lsm, f'Sea-level pressure anomaly (contours: total, 4 hPa), {lab}',
                         contour=msl, contour_levels=np.arange(980, 1041, 4), centres=_centres(p, months)),
               field_map(axes[2, j], z, 'z500', p.lsm, f'500-hPa height anomaly (contours: total, 60 m), {lab}',
                         contour=z_tot, contour_levels=np.arange(5100, 6001, 60))]
    for i, k in enumerate(['sst', 'msl', 'z500']):
        _colorbar(fig, pcs[i], axes[i, :], k)
    return fig


def analog_maps(p, cur_label, cur_months, comp_label, comp, djf_label, djf):
    """Rows: SST, SLP, z500 anomaly; columns: current period, analog composite for the same months, analog DJF."""
    cur = p.mean_anom(cur_months)[:3]
    cols = [(cur_label, cur), (comp_label, comp), (djf_label, djf)]
    fig, axes = plt.subplots(3, 3, figsize=(22, 13), facecolor=SURF, layout='constrained')
    for j, (lab, (sst, z, msl)) in enumerate(cols):
        pcs = [field_map(axes[0, j], sst, 'sst', p.lsm, f'SST anomaly: {lab}'),
               field_map(axes[1, j], msl, 'msl', p.lsm, f'Sea-level pressure anomaly: {lab}'),
               field_map(axes[2, j], z, 'z500', p.lsm, f'500-hPa height anomaly: {lab}')]
    for i, k in enumerate(['sst', 'msl', 'z500']):
        _colorbar(fig, pcs[i], axes[i, :], k)
    return fig


def correlation_maps(p, maps, n, sig):
    fig, axes = plt.subplots(1, 3, figsize=(22, 5.2), facecolor=SURF, layout='constrained')
    titles = ['SST (Nov–Mar)', 'Sea-level pressure (Nov–Mar)', '500-hPa height (Nov–Mar)']
    for ax, m, t in zip(axes, maps, titles):
        pc = field_map(ax, m, 'r', p.lsm, f'{t} vs Flagstaff Nov–Mar precipitation, n = {n}', sig=sig)
    _colorbar(fig, pc, axes, 'r')
    return fig


def index_timeseries(p, start):
    """Interactive stacked bars of the current-state indices (Plotly)."""
    from .interactive import stacked_bars
    f = p.flg_monthly
    oni = p.oni.loc[start:]
    season = pd.Series({t: f'{SEASONS[t.month]} {t.year}' for t in oni.index})
    panels = [
        dict(series=oni, label='ONI', unit='°C', refs=[0.5, 1.0, 1.5, 2.0], custom=season),
        dict(series=p.pdo.loc[start:], label='PDO', unit=''),
        dict(series=p.nph.p_anom.loc[start:], label='North Pacific High\ncentral pressure anomaly', unit='hPa'),
        dict(series=p.jet['A10'].pcs.PC2.loc[start:], label='North Pacific jet\n(Athanasiadis P2)', unit=''),
        dict(series=(f.tavg - f.tavg_n).loc[start:], label='Flagstaff temperature\nanomaly', unit='°F', fmt='+.1f'),
        dict(series=(f.pcpn - f.pcpn_n).loc[start:], label='Flagstaff precipitation\nanomaly', unit='in'),
    ]
    return stacked_bars(panels)


# ---- North Pacific jet ----------------------------------------------------------------------------
def jet_eof_maps(p):
    """Regression of 250-hPa zonal wind on the standardized P2 PC, over the climatological jet of its season."""
    from ..jet import DEFS
    panels = [('A10', 2, 'Athanasiadis P2 (+: southward-displaced subtropical jet)')]
    fig, axes = plt.subplots(1, 1, figsize=(10, 5.6), facecolor=SURF, layout='constrained', squeeze=False)
    levels = np.arange(-10, 10.1, 2)
    norm = BoundaryNorm(levels, DIV.N, extend='both')
    for ax, (k, mode, title) in zip(np.atleast_1d(axes).flat, panels):
        d, e = DEFS[k], p.jet[k]
        pat = e.patterns.sel(mode=mode)
        pc = ax.pcolormesh(pat.longitude, pat.latitude, pat, cmap=DIV, norm=norm, shading='auto')
        u = p.atm.u250
        clim = u.isel(time=np.isin(pd.DatetimeIndex(u.time.values).month, d['months'])).sel(
            time=slice(f"{d['ref'][0]}", f"{d['ref'][1]}")).mean('time')
        cs = ax.contour(clim.longitude, clim.latitude, clim, [30, 40, 50, 60], colors=INK, linewidths=0.7)
        ax.clabel(cs, fontsize=6.5, fmt='%d')
        ax.contour(p.lsm.longitude, p.lsm.latitude, p.lsm, [0.5], colors='#52514e', linewidths=0.5)
        ax.add_patch(plt.Rectangle((d['lon'][0], d['lat'][0]), d['lon'][1] - d['lon'][0], d['lat'][1] - d['lat'][0],
                                   fill=False, ec=MUTED, lw=1, ls=':'))
        ax.plot(360 - 111.65, 35.2, marker='*', color=INK, ms=10, mec=SURF, zorder=6)
        ax.set_xlim(100, 270)
        ax.set_ylim(-5, 80)
        ax.set_aspect(1 / np.cos(np.deg2rad(40)))
        xt = np.arange(120, 271, 30)
        ax.set_xticks(xt, [f'{x}°E' if x < 180 else ('180°' if x == 180 else f'{360 - x}°W') for x in xt])
        ax.tick_params(colors=MUTED, labelsize=7.5)
        var = 100 * e.var_frac[mode - 1]
        ax.set_title(f'{title}; {var:.0f}% of monthly variance', loc='left', fontsize=10, color=INK)
    cb = fig.colorbar(pc, ax=axes, shrink=0.8, aspect=30, pad=0.01)
    cb.set_label('250-hPa zonal wind per standard deviation of PC (m s⁻¹)', color=INK)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    return fig


def jet_winter_series(d):
    """Interactive stacked bars by winter (Plotly)."""
    from .interactive import stacked_bars
    panels = [
        dict(series=d.precip, label='Flagstaff Nov–Mar\nprecipitation', unit='in', colors='median'),
        dict(series=d['A10 P2'], label='Athanasiadis P2\n(Nov–Mar mean)', unit=''),
        dict(series=d['SW trough'], label='Southwest trough\n500-hPa anomaly', unit='m', fmt='+.0f'),
    ]
    return stacked_bars(panels, panel_height=160)
