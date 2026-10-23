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
    al, nph = p.al.loc[months].mean(), p.nph.loc[months].mean()
    return {'L': (al.lat, al.lon), 'H': (nph.lat, nph.lon)}


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
    series = [
        ('ONI (°C)', p.oni, [0.5, 1.0, 1.5, 2.0]),
        ('PDO', p.pdo, []),
        ('PNA', p.pna, []),
        ('North Pacific Index anomaly (hPa)\nnegative = deeper Aleutian Low', p.npi_a, []),
        ('North Pacific High central\npressure anomaly (hPa)', p.nph.p_anom, []),
    ]
    fig, axes = plt.subplots(len(series), 1, figsize=(11, 2.0 * len(series) + 0.6), sharex=True, facecolor=SURF)
    for ax, (lab, s, refs) in zip(axes, series):
        s = s.loc[start:]
        ax.bar(s.index, s.values, width=25, color=[WARM if v >= 0 else COOL for v in s.values], align='edge')
        ax.axhline(0, color=MUTED, lw=0.8)
        for r in refs:
            for sign in (1, -1):
                ax.axhline(sign * r, color=GRID, lw=0.8, ls='--', zorder=0)
        ax.set_ylabel(lab, fontsize=8.5)
        clean(ax)
        t, last = s.dropna().index[-1], s.dropna().iloc[-1]
        when = (f'{SEASONS[t.month]} {t.year}' if lab.startswith('ONI') else f'{t:%b %Y}')
        ax.annotate(f'{when}: {last:+.2f}', xy=(1, 1), xycoords='axes fraction', ha='right',
                    va='bottom', fontsize=8.5, color=INK)
    loc = plt.matplotlib.dates.AutoDateLocator(maxticks=10)
    axes[-1].xaxis.set_major_locator(loc)
    axes[-1].xaxis.set_major_formatter(plt.matplotlib.dates.ConciseDateFormatter(loc))
    fig.tight_layout()
    return fig


# ---- North Pacific jet ----------------------------------------------------------------------------
def jet_eof_maps(p):
    """Regression of 250-hPa zonal wind on each standardized PC, over the climatological jet of its season."""
    from ..jet import DEFS
    panels = [('A10', 1, 'Athanasiadis P1 (≈ PNA)'), ('A10', 2, 'Athanasiadis P2 (+: southward-displaced subtropical jet)'),
              ('W19', 1, 'NPJ EOF1 (+: jet extension)'), ('W19', 2, 'NPJ EOF2 (+: poleward shift)')]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor=SURF, layout='constrained')
    levels = np.arange(-10, 10.1, 2)
    norm = BoundaryNorm(levels, DIV.N, extend='both')
    for ax, (k, mode, title) in zip(axes.flat, panels):
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


def npj_phase_diagram(p, n_months=12):
    pcs = p.jet['W19'].pcs.dropna().iloc[-n_months:]
    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor=SURF)
    lim = max(3.0, float(np.abs(pcs.values).max()) + 0.5)
    ax.add_patch(plt.Circle((0, 0), 1, fill=True, fc='#f0efec', ec=MUTED, lw=0.8, zorder=0))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.plot(pcs.PC1, pcs.PC2, color=MUTED, lw=1, zorder=2)
    season = np.isin(pcs.index.month, [9, 10, 11, 12, 1, 2, 3, 4, 5])
    ax.scatter(pcs.PC1[season], pcs.PC2[season], s=60, color=BLUE_ACC, edgecolor=INK, lw=0.6, zorder=3,
               label='Sep–May (season of the Winters et al. EOFs)')
    ax.scatter(pcs.PC1[~season], pcs.PC2[~season], s=60, facecolor=SURF, edgecolor=INK, lw=0.8, zorder=3,
               label='Jun–Aug (outside that season)')
    for t, r in pcs.iterrows():
        ax.annotate(f'{t:%b %y}', (r.PC1, r.PC2), xytext=(6, 4), textcoords='offset points', fontsize=8, color=INK)
    last = pcs.iloc[-1]
    ax.scatter([last.PC1], [last.PC2], s=150, facecolor='none', edgecolor=WARM, lw=2, zorder=4, label='latest month')
    for txt, x, y, ha, va in [('Jet extension', lim * 0.95, 0.1, 'right', 'bottom'),
                              ('Jet retraction', -lim * 0.95, 0.1, 'left', 'bottom'),
                              ('Poleward shift', 0.1, lim * 0.95, 'left', 'top'),
                              ('Equatorward shift', 0.1, -lim * 0.95, 'left', 'bottom')]:
        ax.text(x, y, txt, ha=ha, va=va, fontsize=9, color=MUTED, style='italic')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('NPJ PC1 (+: extension)')
    ax.set_ylabel('NPJ PC2 (+: poleward shift)')
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    clean(ax)
    fig.tight_layout()
    return fig


def jet_winter_series(d):
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True, facecolor=SURF)
    pr = d.precip
    axes[0].bar(pr.index, pr.values, color=[WARM if v >= pr.median() else COOL for v in pr.values], width=0.8)
    axes[0].axhline(pr.median(), color=MUTED, lw=0.8, ls='--')
    axes[0].set_ylabel('Flagstaff Nov–Mar\nprecipitation (in)', fontsize=8.5)
    for ax, k, lab in [(axes[1], 'A10 P2', 'Athanasiadis P2\n(Nov–Mar mean PC)'),
                       (axes[2], 'SW trough', 'Southwest trough\n500-hPa anomaly (m)')]:
        s = d[k]
        ax.bar(s.index, s.values, color=[WARM if v >= 0 else COOL for v in s.values], width=0.8)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylabel(lab, fontsize=8.5)
    for ax in axes:
        clean(ax)
    axes[-1].set_xlabel('Winter (starting year)')
    fig.tight_layout()
    return fig
