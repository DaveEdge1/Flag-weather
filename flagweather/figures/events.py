"""Event figures: ERA5 synoptic maps and network rain vs. ERA5 time series at Flagstaff."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import AR_IVT, FLAGSTAFF
from .style import BLUE, BLUE_LIGHT, GRID, INK, IVT_CMAP, MUTED, ORANGE, SURF, clean


def point_series(ds):
    """ERA5 at the grid point nearest Flagstaff."""
    p = ds.sel(latitude=FLAGSTAFF['lat'], longitude=FLAGSTAFF['lon'], method='nearest')
    df = p[['ivt', 'tcwv', 'ws250', 'z500']].to_dataframe()[['ivt', 'tcwv', 'ws250', 'z500']]
    df['ivt_from_deg'] = (np.degrees(np.arctan2(-p.ivt_u, -p.ivt_v)) % 360).values
    df.attrs['grid_point'] = (float(p.latitude), float(p.longitude))
    return df


def network_timeseries(hourly, df, title=None):
    """Hourly network rain (median and 10-90th percentile across gauges) above ERA5 IVT, 250-hPa wind, 500-hPa height."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True, facecolor=SURF,
                             gridspec_kw={'height_ratios': [1.2, 1, 1, 1]})
    h = hourly.loc[df.index[0]:df.index[-1] + pd.Timedelta('6h')]
    q10, q50, q90 = (h.quantile(q, axis=1) for q in (0.1, 0.5, 0.9))
    axes[0].fill_between(h.index, q10, q90, step='post', color=BLUE_LIGHT, lw=0, label='10–90th percentile')
    axes[0].step(h.index, q50, where='post', color=BLUE, lw=1.5, label='median')
    axes[0].legend(frameon=False, fontsize=8.5, loc='upper left')
    axes[0].set_ylabel(f'Rain, {h.shape[1]} city\ngauges (in/h)')
    axes[1].plot(df.index, df.ivt, color=INK, lw=2)
    axes[1].axhline(AR_IVT, color=ORANGE, lw=1.5, ls='--')
    axes[1].text(df.index[0], AR_IVT + 15, f'{AR_IVT} kg m⁻¹ s⁻¹', color=ORANGE, fontsize=8.5)
    axes[1].set_ylabel('IVT\n(kg m⁻¹ s⁻¹)')
    axes[2].plot(df.index, df.ws250, color=INK, lw=2)
    axes[2].set_ylabel('250-hPa wind\n(m s⁻¹)')
    axes[3].plot(df.index, df.z500, color=INK, lw=2)
    axes[3].set_ylabel('500-hPa height\n(m)')
    loc = plt.matplotlib.dates.AutoDateLocator(maxticks=10)
    axes[3].xaxis.set_major_locator(loc)
    axes[3].xaxis.set_major_formatter(plt.matplotlib.dates.ConciseDateFormatter(loc))
    for ax in axes:
        clean(ax)
    lat, lon = df.attrs.get('grid_point', (np.nan, np.nan))
    axes[0].set_title(title or f'City gauge network rain and ERA5 at {lat:.2f}°N, {abs(lon):.2f}°W (UTC)',
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
    ax.quiver(q.longitude, q.latitude, q.ivt_u.where(m), q.ivt_v.where(m), color=INK, scale=12000, width=0.003)
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
    ax.set_title(pd.Timestamp(t).strftime('%b %d %H UTC'), loc='left', fontsize=10, color=INK)
    return pc


def map_panels(ds, lsm, times, best_track=None, ncols=3):
    nrows = int(np.ceil(len(times) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.6 * nrows + 1.0), facecolor=SURF, squeeze=False)
    for ax, t in zip(axes.flat, times):
        pc = draw_map(ax, ds, lsm, t, best_track)
    for ax in axes.flat[len(times):]:
        ax.set_visible(False)
    cb = fig.colorbar(pc, ax=axes, orientation='horizontal', fraction=0.03, pad=0.03, aspect=50)
    cb.set_label('IVT (kg m⁻¹ s⁻¹); blue contour and arrows: IVT ≥ 250', color=INK)
    fig.suptitle('ERA5: IVT (shading), 500-hPa height (black, 60 m), 250-hPa wind 40/55/70 m s⁻¹ (orange); '
                 '★ Flagstaff; ✕ NHC best-track position', x=0.01, ha='left', fontsize=12, color=INK)
    return fig
