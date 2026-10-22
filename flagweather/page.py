"""Text and tables for event pages (observations only, from computed numbers)."""
import numpy as np
import pandas as pd

from .config import AR_IVT, TC_RADIUS_KM
from .event import JET_RADIUS_KM, LOW_RADIUS_KM, TROUGH_WEST

MST = pd.Timedelta('7h')
COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
TC_TYPES = {'HU': 'hurricane', 'TS': 'tropical storm', 'TD': 'tropical depression', 'DB': 'remnant disturbance',
            'LO': 'remnant low', 'EX': 'extratropical remnant', 'SD': 'subtropical depression', 'SS': 'subtropical storm'}


def _t(t):
    t = pd.Timestamp(t) - MST
    return f'{t:%b} {t.day} {t:%H:%M} MST'


def _dir(deg):
    return COMPASS[int((deg % 360) / 22.5 + 0.5) % 16]


def _hours(a, b):
    return (pd.Timestamp(b) - pd.Timestamp(a)) / pd.Timedelta('1h')


def _ordinal(n):
    return f'{n}{"th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")}'


def _signed(x, unit='°F'):
    return f'{x:+.1f} {unit}'


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


def narrative(ev):
    """Event summary emphasising the atmospheric setting. Each sentence states a computed quantity."""
    n = ev.network_stats()
    t0, t1 = ev.rain_core()
    out = [f'Across {n["n"]} gauges the event total ranged from {n["min"]:.2f} to {n["max"]:.2f} in '
           f'(median {n["median"]:.2f} in). The central 80% of the network-mean rain fell between {_t(t0)} and '
           f'{_t(t1)} ({_hours(t0, t1):.0f} h).']
    s = ev.atmos_setting()
    if s is None:
        for _, r in ev.tropical_cyclones().iterrows():
            out.append(f'{r["name"]} ({TC_TYPES.get(r.type, r.type)} at the time) was {r.km:,.0f} km from '
                       f'Flagstaff at {_t(r.time)}.')
        out.append('ERA5 does not yet cover this event, so the atmospheric setting is not yet described.')
        return ' '.join(out)
    e = ev.era5_stats()
    if e['ivt_max'] >= AR_IVT:
        out.append(f'Moisture transport over Flagstaff reached atmospheric-river strength: ERA5 IVT was at or above '
                   f'{AR_IVT} kg m⁻¹ s⁻¹ for {e["ar_hours"]} h and peaked at {e["ivt_max"]:.0f} kg m⁻¹ s⁻¹ from '
                   f'the {_dir(e["ivt_dir"])} at {_t(e["ivt_time"])}. {100 * s["frac_ar"]:.0f}% of the '
                   f'network-mean rain fell while IVT was at or above {AR_IVT}.')
    else:
        out.append(f'Moisture transport over Flagstaff stayed below atmospheric-river strength: ERA5 IVT peaked at '
                   f'{e["ivt_max"]:.0f} kg m⁻¹ s⁻¹ from the {_dir(e["ivt_dir"])} at {_t(e["ivt_time"])}, and total '
                   f'column water vapour reached {e["tcwv_max"]:.0f} mm.')
    for c in s['tc_connected']:
        out.append(f'At the IVT peak, the region of IVT ≥ {AR_IVT} over Flagstaff was continuous with the area '
                   f'around {c["name"]} ({TC_TYPES.get(c["type"], c["type"])}, {c["lat"]:.1f}°N '
                   f'{abs(c["lon"]):.1f}°W, {c["km"]:,.0f} km to the {_dir(c["bearing"])}).')
    tc = ev.tropical_cyclones()
    if not tc.empty:
        for _, r in tc[~tc['name'].isin([c['name'] for c in s['tc_connected']])].iterrows():
            out.append(f'{r["name"]} came within {r.km:,.0f} km of Flagstaff ({_t(r.time)}, '
                       f'{TC_TYPES.get(r.type, r.type)}).')
    lat = f'{ev.point.attrs["grid_point"][0]:.2f}°N'
    if s['trough_at_flg']:
        trough = (f'the lowest 500-hPa height along {lat} between {abs(TROUGH_WEST)}°W and Flagstaff was over '
                  f'Flagstaff itself ({s["z500_flg"]:,.0f} m)')
    elif s['trough_lon'] is None:
        trough = f'there was no 500-hPa trough axis along {lat} between {abs(TROUGH_WEST)}°W and Flagstaff'
    else:
        trough = (f'the 500-hPa trough axis along {lat} was at {abs(s["trough_lon"]):.1f}°W, {s["trough_km"]:,.0f} km '
                  f'west of Flagstaff ({s["trough_z"]:,.0f} m, against {s["z500_flg"]:,.0f} m over Flagstaff)')
    out.append(f'At the IVT peak, 500-hPa flow over Flagstaff was from the {_dir(s["flow500_from"])} at '
               f'{s["flow500_speed"]:.0f} m s⁻¹, and {trough}.')
    cl = s['closed_low']
    if cl:
        out.append(f'A closed 500-hPa low ({cl["z"]:,.0f} m) was centred {cl["km"]:,.0f} km to the '
                   f'{_dir(cl["bearing"])}, at {cl["lat"]:.1f}°N {abs(cl["lon"]):.1f}°W.')
    else:
        out.append(f'There was no closed 500-hPa low within {LOW_RADIUS_KM:,} km.')
    out.append(f'The strongest 250-hPa wind within {JET_RADIUS_KM:,} km was {s["jet_speed"]:.0f} m s⁻¹, '
               f'{s["jet_km"]:,.0f} km to the {_dir(s["jet_bearing"])}.')
    dh = _hours(e['ivt_time'], s['ws250_flg_time'])
    when = 'at the same time as' if abs(dh) < 1 else f'{abs(dh):.0f} h {"after" if dh > 0 else "before"}'
    out.append(f'The 250-hPa wind over Flagstaff peaked at {s["ws250_flg_max"]:.0f} m s⁻¹ ({_t(s["ws250_flg_time"])}), '
               f'{when} the IVT peak.')
    return ' '.join(out)


def _vs(obs, ref, label):
    """'x in (p% of label)' when the reference is at least 2 in, otherwise 'x in (label y in)'."""
    if pd.notna(ref) and ref >= 2:
        return f'{obs:.1f} in ({100 * obs / ref:.0f}% of {label})'
    return f'{obs:.2f} in ({label} {ref:.2f} in)' if pd.notna(ref) else f'{obs:.2f} in'


def climate(ev):
    a, sn = ev.airport_stats(), ev.snotel_stats()
    ratio = a['precip'] / a['precip_normal'] if a['precip_normal'] else np.nan
    days = f'{a["ndays"]}-day'
    end = ev.days[-1]
    wy0 = pd.Timestamp(end.year - (end.month < 10), 10, 1)
    miss = f' ({a["missing"]} day{"s" * (a["missing"] > 1)} not yet reported)' if a['missing'] else ''
    lines = [
        f'- **Airport precipitation:** {a["precip"]:.2f} in over {a["ndays"]} local day{"s" * (a["ndays"] > 1)}'
        f'{miss}, against a 1991–2020 normal of {a["precip_normal"]:.2f} in for those days ({ratio:.0f}× normal).',
        f'- **Rank:** the {_ordinal(a["rank"])} largest {days} airport total in {end:%B} among {a["nyears"]} years '
        f'with data since {a["first_year"]} (each year\'s largest {days} total ending in {end:%B}).',
        f'- **Airport temperature:** highs averaged {_signed(a["tmax_dep"])} and lows {_signed(a["tmin_dep"])} '
        f'relative to normal.',
        f'- **Water year to date** ({wy0:%b} {wy0.day}, {wy0.year} – {end:%b} {end.day}, {end.year}): airport '
        + _vs(a['wytd'], a['wytd_normal'], 'normal') + '; SNOTEL '
        + '; '.join(f'{k} ' + _vs(r.wy, r.wy_median, 'median') for k, r in sn.iterrows()) + '.',
    ]
    return '\n'.join(lines)


def key_numbers(ev):
    n, sn = ev.network_stats(), ev.snotel_stats()
    lines = [
        f'- **Network rain ({n["n"]} gauges):** median {n["median"]:.2f} in; 10th–90th percentile '
        f'{n["p10"]:.2f}–{n["p90"]:.2f} in; lowest {n["min"]:.2f} in ({n["min_site"]}), highest {n["max"]:.2f} in '
        f'({n["max_site"]}).',
        f'- **Peak hour:** network median {n["peak_hour_median"]:.2f} in (hour beginning {_t(n["peak_hour"])}); '
        f'highest single-gauge hour {n["peak_hour_max"]:.2f} in.',
        '- **SNOTEL event precipitation:** ' + '; '.join(f'{k} {r.precip:.1f} in' for k, r in sn.iterrows()) + '.',
    ]
    e = ev.era5_stats()
    if e is not None:
        lines.append(f'- **ERA5 at Flagstaff:** peak IVT {e["ivt_max"]:.0f} kg m⁻¹ s⁻¹; maximum 250-hPa wind '
                     f'{e["ws250_max"]:.0f} m s⁻¹; minimum 500-hPa height {e["z500_min"]:,.0f} m; maximum total '
                     f'column water vapour {e["tcwv_max"]:.0f} mm.')
    for _, r in ev.tropical_cyclones().iterrows():
        lines.append(f'- **{r["name"]}** (NHC best track, peak {r.peak_kt} kt): closest approach {r.km:,.0f} km at '
                     f'{_t(r.time)}, {r.lat:.1f}°N {abs(r.lon):.1f}°W ({TC_TYPES.get(r.type, r.type)}, {r.vmax_kt} kt).')
    if not ev.final:
        lines.append('- **Provisional:** ERA5 does not yet cover this event; maps will be added when it does.')
    return '\n'.join(lines)


def station_table(ev):
    a = ev.airport.copy()
    a.index = [f'{d:%b} {d.day}' for d in a.index]
    def f(fmt):   # 'M' marks values not yet reported
        return lambda v: 'M' if pd.isna(v) else fmt.format(v)
    t = pd.DataFrame({
        'Precip (in)': a.pcpn.map(f('{:.2f}')), 'Normal (in)': a.pcpn_n.map(f('{:.2f}')),
        'Snow (in)': a.snow.map(f('{:.1f}')),
        'High (°F)': a.maxt.map(f('{:.0f}')), 'High vs normal': (a.maxt - a.maxt_n).map(f('{:+.0f}')),
        'Low (°F)': a.mint.map(f('{:.0f}')), 'Low vs normal': (a.mint - a.mint_n).map(f('{:+.0f}')),
    })
    tot = a[['pcpn', 'pcpn_n', 'snow']].sum()
    t.loc['Total'] = [f'{tot.pcpn:.2f}', f'{tot.pcpn_n:.2f}', f'{tot.snow:.1f}', '', '', '', '']
    return t.to_markdown()


def snotel_table(ev):
    s = ev.snotel_stats()
    t = pd.DataFrame({
        'Event precip (in)': s.precip.map('{:.1f}'.format),
        'Water-year precip (in)': s.wy.map('{:.1f}'.format),
        'Median to date (in)': s.wy_median.map('{:.1f}'.format),
        '% of median': [f'{p:.0f}' if pd.notna(p) and m >= 2 else '—' for p, m in zip(s.wy_pct, s.wy_median)],
        'SWE at end (in)': s.swe_end.map('{:.1f}'.format),
        'Max snow depth (in)': s.snwd_max.map('{:.0f}'.format),
    })
    return t.to_markdown()
