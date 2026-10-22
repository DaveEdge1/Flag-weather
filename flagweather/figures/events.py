"""Event figures: ERA5 synoptic maps and network rain vs. ERA5 time series at Flagstaff."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects

from .. import basemap
from ..config import AR_IVT, FLAGSTAFF
from .style import BLUE, BLUE_LIGHT, GRID, INK, IVT_CMAP, MUTED, ORANGE, SURF, clean


def point_series(ds):
    """ERA5 at the grid point nearest Flagstaff."""
    p = ds.sel(latitude=FLAGSTAFF['lat'], longitude=FLAGSTAFF['lon'], method='nearest')
    df = p[['ivt', 'tcwv', 'ws250', 'z500']].to_dataframe()[['ivt', 'tcwv', 'ws250', 'z500']]
    df['ivt_from_deg'] = (np.degrees(np.arctan2(-p.ivt_u, -p.ivt_v)) % 360).values
    df.attrs['grid_point'] = (float(p.latitude), float(p.longitude))
    return df


def gauge_map(net, values, title, snotel_sites=None, snotel_values=None, unit='in'):
    """Gauges (circle area proportional to value) over shaded relief, streams and highways;
    SNOTEL triangles; airport square."""
    fig, ax = plt.subplots(figsize=(10, 9), facecolor=SURF)
    basemap.draw(ax)
    halo = [patheffects.withStroke(linewidth=2.2, foreground=SURF)]
    v = values.reindex(net.index)
    scale = 700 / max(float(v.max()), 0.01)
    ax.scatter(net.lon, net.lat, s=v.fillna(0) * scale, color=BLUE, alpha=0.7, edgecolor=INK, lw=0.5, zorder=3)
    for gid, r in net.iterrows():
        if pd.notna(v[gid]):
            ax.annotate(f'{v[gid]:.2f}', (r.lon, r.lat), fontsize=7, color=INK, ha='center', va='center', zorder=4,
                        path_effects=halo)
    if snotel_sites is not None:
        ax.scatter(snotel_sites.lon, snotel_sites.lat, marker='^', s=110, color=ORANGE, edgecolor=INK, lw=0.7, zorder=4)
        for t, r in snotel_sites.iterrows():
            lab = r['name'] if snotel_values is None or pd.isna(snotel_values.get(r['name']))                 else f"{r['name']} {snotel_values[r['name']]:.1f}"
            ax.annotate(lab, (r.lon, r.lat), xytext=(-8, 4), textcoords='offset points', fontsize=8.5, color=INK,
                        ha='right', zorder=4, path_effects=halo)
    ax.scatter([-111.666], [35.144], marker='s', s=60, color=INK, zorder=4)
    ax.annotate('Airport', (-111.666, 35.144), xytext=(6, -10), textcoords='offset points', fontsize=8.5, color=INK,
                zorder=4, path_effects=halo)
    ax.set_aspect(1 / np.cos(np.deg2rad(35.2)))
    ax.set_title(title, loc='left', color=INK)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    fig.tight_layout()
    return fig


def _rain_panel(ax, h):
    q10, q50, q90 = (h.quantile(q, axis=1) for q in (0.1, 0.5, 0.9))
    ax.fill_between(h.index, q10, q90, step='post', color=BLUE_LIGHT, lw=0, label='10–90th percentile')
    ax.step(h.index, q50, where='post', color=BLUE, lw=1.5, label='median')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    ax.set_ylabel(f'Hourly rain,\n{h.shape[1]} gauges (in)')


def _cum_panel(ax, h):
    c = h.cumsum()
    q10, q50, q90 = (c.quantile(q, axis=1) for q in (0.1, 0.5, 0.9))
    ax.fill_between(c.index, q10, q90, color=BLUE_LIGHT, lw=0, label='10–90th percentile')
    ax.plot(c.index, q50, color=BLUE, lw=2, label='median')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    ax.set_ylabel('Cumulative\nrain (in)')


def _time_axis(ax):
    loc = plt.matplotlib.dates.AutoDateLocator(maxticks=10)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.ConciseDateFormatter(loc))


def network_timeseries(hourly, df=None, title=None, start=None, end=None):
    """Hourly and cumulative network rain (median and 10-90th percentile across gauges); with df (ERA5 point
    series), also IVT, 250-hPa wind and 500-hPa height at Flagstaff. Cumulative rain starts at `start`."""
    if df is None:
        fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, facecolor=SURF)
        h = hourly.loc[start:end]
        _rain_panel(axes[0], h)
        _cum_panel(axes[1], h)
        _time_axis(axes[1])
        for ax in axes:
            clean(ax)
        axes[0].set_title(title or 'Gauge network rain (UTC)', loc='left', fontsize=11.5, color=INK)
        fig.tight_layout()
        return fig
    fig, axes = plt.subplots(5, 1, figsize=(11, 10.5), sharex=True, facecolor=SURF,
                             gridspec_kw={'height_ratios': [1.2, 1, 1, 1, 1]})
    h = hourly.loc[df.index[0]:df.index[-1] + pd.Timedelta('6h')]
    _rain_panel(axes[0], h)
    _cum_panel(axes[1], hourly.loc[start or df.index[0]:end or h.index[-1]])
    axes[2].plot(df.index, df.ivt, color=INK, lw=2)
    axes[2].axhline(AR_IVT, color=ORANGE, lw=1.5, ls='--')
    axes[2].text(df.index[0], AR_IVT + 15, f'{AR_IVT} kg m⁻¹ s⁻¹', color=ORANGE, fontsize=8.5)
    axes[2].set_ylabel('IVT\n(kg m⁻¹ s⁻¹)')
    axes[3].plot(df.index, df.ws250, color=INK, lw=2)
    axes[3].set_ylabel('250-hPa wind\n(m s⁻¹)')
    axes[4].plot(df.index, df.z500, color=INK, lw=2)
    axes[4].set_ylabel('500-hPa height\n(m)')
    _time_axis(axes[4])
    for ax in axes:
        clean(ax)
    lat, lon = df.attrs.get('grid_point', (np.nan, np.nan))
    axes[0].set_title(title or f'Gauge network rain and ERA5 at {lat:.2f}°N, {abs(lon):.2f}°W (UTC)',
                      loc='left', fontsize=11.5, color=INK)
    fig.tight_layout()
    return fig


def draw_map(ax, ds, lsm, t, best_track=None):
    s = ds.sel(valid_time=t)
    lon, lat = s.longitude, s.latitude
    pc = ax.pcolormesh(lon, lat, s.ivt.where(s.ivt >= 100), cmap=IVT_CMAP, vmin=100, vmax=800, shading='auto')
    ax.contour(lon, lat, s.ivt, [AR_IVT], colors=BLUE, linewidths=1.2)
    ax.contour(lsm.longitude, lsm.latitude, lsm, [0.5], colors=MUTED, linewidths=0.6)
    cz = ax.contour(lon, lat, s.z500, np.arange(5100, 6000, 60), colors=INK, linewidths=0.6)
    ax.clabel(cz, cz.levels[::2], fontsize=6.5, fmt='%d')
    ax.contour(lon, lat, s.ws250, [40, 55, 70], colors=ORANGE, linewidths=[0.9, 1.5, 2.2])
    q = s.isel(latitude=slice(None, None, 12), longitude=slice(None, None, 12))  # every 3 degrees
    m = q.ivt >= AR_IVT
    ax.quiver(q.longitude, q.latitude, q.ivt_u.where(m), q.ivt_v.where(m), color=INK, scale=14000, width=0.0018)
    if best_track is not None:
        bt = best_track[best_track.time == pd.Timestamp(t)]
        inside = bt[(bt.lon > float(lon.min())) & (bt.lon < float(lon.max())) &
                    (bt.lat > float(lat.min())) & (bt.lat < float(lat.max()))]
        for _, r in inside.iterrows():
            ax.plot(r.lon, r.lat, marker='X', color=INK, ms=7, mec=SURF, mew=0.8, zorder=5)
            ax.annotate(r['name'], (r.lon, r.lat), xytext=(4, -9), textcoords='offset points', fontsize=7.5, color=INK)
    ax.plot(FLAGSTAFF['lon'], FLAGSTAFF['lat'], marker='*', color=ORANGE, ms=11, mec=INK, mew=0.6, zorder=5)
    ax.set_xlim(float(lon.min()), float(lon.max()))
    ax.set_ylim(float(lat.min()), float(lat.max()))
    ax.set_aspect(1 / np.cos(np.deg2rad(35)))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    t = pd.Timestamp(t)
    ax.set_title(f"{t:%b %d %H} UTC ({t - pd.Timedelta('7h'):%b %d %H:%M} MST)", loc='left', fontsize=11, color=INK)
    return pc


def map_panels(ds, lsm, times, best_track=None, ncols=1):
    nrows = int(np.ceil(len(times) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 8.3 * nrows + 1.0), facecolor=SURF, squeeze=False,
                             layout='constrained')
    for ax, t in zip(axes.flat, times):
        pc = draw_map(ax, ds, lsm, t, best_track)
    for ax in axes.flat[len(times):]:
        ax.set_visible(False)
    cb = fig.colorbar(pc, ax=axes, orientation='horizontal', shrink=0.7, aspect=45, pad=0.01)
    cb.set_label('IVT (kg m⁻¹ s⁻¹); blue contour and arrows: IVT ≥ 250. Black: 500-hPa height (60 m). '
                 'Orange: 250-hPa wind 40/55/70 m s⁻¹. ★ Flagstaff. ✕ NHC best track.', color=INK)
    return fig
