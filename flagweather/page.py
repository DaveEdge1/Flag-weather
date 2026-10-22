"""Text and tables for event pages (observations only, from computed numbers)."""
import pandas as pd

from .config import AR_IVT, TC_RADIUS_KM

MST = pd.Timedelta('7h')
COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']


def _fmt_time(t):
    t = pd.Timestamp(t) - MST
    return f'{t:%b} {t.day} {t:%H:%M} MST'


def _dir(deg):
    return COMPASS[int((deg % 360) / 22.5 + 0.5) % 16]


def description(ev):
    n = ev.network_stats()
    s = f'Network median {n["median"]:.2f} in (range {n["min"]:.2f}–{n["max"]:.2f} in)'
    e = ev.era5_stats()
    if e is not None:
        s += f'; peak IVT {e["ivt_max"]:.0f} kg m⁻¹ s⁻¹ from the {_dir(e["ivt_dir"])}'
    tc = ev.tropical_cyclones()
    if not tc.empty:
        s += '; ' + ', '.join(tc['name']) + ' within ' + f'{TC_RADIUS_KM:,} km'
    return s


def summary(ev):
    n, a, sn = ev.network_stats(), ev.airport_stats(), ev.snotel_stats()
    lines = [
        f'- **Rain across {n["n"]} city/county gauges:** median {n["median"]:.2f} in; '
        f'10th–90th percentile {n["p10"]:.2f}–{n["p90"]:.2f} in; '
        f'lowest {n["min"]:.2f} in ({n["min_site"]}), highest {n["max"]:.2f} in ({n["max_site"]}).',
        f'- **Peak hour:** network median {n["peak_hour_median"]:.2f} in (hour beginning {_fmt_time(n["peak_hour"])}); '
        f'highest single-gauge hour {n["peak_hour_max"]:.2f} in.',
        f'- **Airport:** {a["precip"]:.2f} in precipitation, {a["snow"]:.1f} in snow; '
        f'temperature {a["tmin"]:.0f}–{a["tmax"]:.0f} °F.',
        '- **SNOTEL precipitation:** ' + '; '.join(f'{k} {r.precip:.1f} in' for k, r in sn.iterrows())
        + ('. Snow water equivalent at end: ' + '; '.join(f'{k} {r.swe_end:.1f} in' for k, r in sn.iterrows()) + '.'
           if sn.swe_end.max() > 0 else '.'),
    ]
    e = ev.era5_stats()
    if e is not None:
        lines.append(f'- **ERA5 at Flagstaff:** IVT peaked at {e["ivt_max"]:.0f} kg m⁻¹ s⁻¹ from the {_dir(e["ivt_dir"])} '
                     f'({_fmt_time(e["ivt_time"])}); ' + (f'IVT ≥ {AR_IVT} for {e["ar_hours"]} h; ' if e['ar_hours'] else '') +
                     f'maximum 250-hPa wind {e["ws250_max"]:.0f} m s⁻¹; minimum 500-hPa height {e["z500_min"]:.0f} m.')
    tc = ev.tropical_cyclones()
    for _, r in tc.iterrows():
        lines.append(f'- **{r["name"]}** (NHC best track; peak {r.peak_kt} kt) came within {r.km:,.0f} km of Flagstaff '
                     f'at {_fmt_time(r.time)}, at {r.lat:.1f}°N {abs(r.lon):.1f}°W ({r.type}, {r.vmax_kt} kt).')
    if not ev.final:
        lines.append('- **Provisional:** ERA5 does not yet cover this event; maps will be added when it does.')
    return '\n'.join(lines)


def station_table(ev):
    a = ev.airport.copy()
    a.index = a.index.strftime('%b %d')
    t = a[['pcpn', 'snow', 'maxt', 'mint']].rename(columns={'pcpn': 'Precip (in)', 'snow': 'Snow (in)',
                                                           'maxt': 'High (°F)', 'mint': 'Low (°F)'})
    return t.to_markdown()


def snotel_table(ev):
    s = ev.snotel_stats().rename(columns={'precip': 'Precip (in)', 'swe_end': 'SWE at end (in)',
                                          'snwd_max': 'Max snow depth (in)', 'tmin': 'Low (°F)', 'tmax': 'High (°F)'})
    return s.round(1).to_markdown()
