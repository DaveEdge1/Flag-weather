"""Text and tables for event pages (observations only, from computed numbers)."""
import numpy as np
import pandas as pd

from .config import AR_IVT, ROOT, TC_RADIUS_KM
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


WORDS = {'N': 'north', 'S': 'south', 'E': 'east', 'W': 'west'}


def _dir_words(deg):
    """Compass point in words, e.g. 'west-northwest'."""
    c = _dir(deg)
    if len(c) == 3:
        return f'{WORDS[c[0]]}-{WORDS[c[1]]}{WORDS[c[2]]}'
    return ''.join(WORDS[x] for x in c)


def _mi(km):
    return f'{round(km * 0.621371, -1):,.0f}'


def _part(t):
    h = (pd.Timestamp(t) - MST).hour
    return ('overnight' if h < 5 else 'morning' if h < 12 else 'afternoon' if h < 17
            else 'evening' if h < 21 else 'night')


def _span(t0, t1):
    """Local (MST) span, e.g. 'Saturday night, October 31, and Sunday evening, November 1'."""
    a, b = pd.Timestamp(t0) - MST, pd.Timestamp(t1) - MST
    def day(x, t):
        p = _part(t)
        return f'{p} {x:%A}' if p == 'overnight' else f'{x:%A} {p}'
    if a.date() == b.date():
        return f'{day(a, t0)} and {_part(t1)}, {a:%B} {a.day}'
    return f'{day(a, t0)}, {a:%B} {a.day}, and {day(b, t1)}, {b:%B} {b.day}'


def _tc_name(type_, name):
    if type_ in ('DB', 'LO', 'EX'):
        return f'the remnants of {name}'
    return f'{TC_TYPES.get(type_, type_).title()} {name}'


def plain_summary(ev):
    """The event in plain language: when, how much, how unusual, and what caused it."""
    n, a, sn = ev.network_stats(), ev.airport_stats(), ev.snotel_stats()
    t0, t1 = ev.rain_core()
    out = [f'Most of the rain fell between {_span(t0, t1)}, over about {_hours(t0, t1):.0f} hours.',
           f'Rain gauges across Flagstaff caught a typical {n["median"]:.2f} inches, from {n["min"]:.2f} in at '
           f'{n["min_site"]} to {n["max"]:.2f} in at {n["max_site"]}.']
    days = f'{a["ndays"]} day{"s" * (a["ndays"] > 1)}'
    ratio = a['precip'] / a['precip_normal'] if a['precip_normal'] else np.nan
    s = f'The airport measured {a["precip"]:.2f} in'
    if np.isfinite(ratio) and ratio >= 1.5:
        s += f', about {ratio:.0f} times the normal for ' + (f'those {days}' if a['ndays'] > 1 else 'the day')
    s += '.'
    month = f'{ev.days[-1]:%B}'
    if a['rank'] == 1:
        s += f' That is the most for any {days} in {month} at the airport since records began in {a["first_year"]}.'
    elif a['rank'] <= 10:
        s += (f' Only {a["rank"] - 1} {month}{"s" * (a["rank"] > 2)} since {a["first_year"]} '
              f'had a wetter {days}.')
    out.append(s)
    if a['snow'] > 0:
        out.append(f'{a["snow"]:.1f} in of snow fell at the airport.')
    elif sn.snwd_max.max() > 0:
        out.append('Snow fell only at the highest mountain sites; in town it was rain.')
    if a['tmin_dep'] >= 5:
        out.append(f'Nights were mild for the season, about {a["tmin_dep"]:.0f} °F above normal.')
    elif a['tmax_dep'] <= -5:
        out.append(f'Days were cool, about {-a["tmax_dep"]:.0f} °F below normal.')

    lead = setup_text(ev.slug)
    cause = []
    st = ev.atmos_setting()
    if st is None:
        for _, r in ev.tropical_cyclones().iterrows():
            cause.append(f'{_tc_name(r.type, r["name"])[0].upper()}{_tc_name(r.type, r["name"])[1:]} was about '
                         f'{_mi(r.km)} miles away at its closest.')
        cause.append('Weather maps of what caused the storm will be added once the ERA5 reanalysis covers it, '
                     'usually about a week afterwards.')
    elif lead.get('setup'):
        cause = _cause_short(ev, st)
    else:
        cause = _cause_full(ev, st)
    return _paragraphs(lead, out, cause, stream_sentences(ev))


def _streams(ev):
    """Regional USGS peaks and city stage peaks for the event (computed once per event), or None if unavailable."""
    if not hasattr(ev, '_streams'):
        from . import streams
        try:
            near, flag = streams.regional(ev)
            ev._streams = (near, flag, streams.city(ev))
        except Exception:   # a data service was unreachable at build time
            ev._streams = None
    return ev._streams


def stream_sentences(ev):
    """Summary sentences on stream peaks, when gauges ran well above normal (see streams.sentences)."""
    from . import streams
    st = _streams(ev)
    if st is None:
        return []
    return streams.sentences(*st, start=ev.start.tz_localize(None) if ev.start.tzinfo else ev.start)


def streamflow(ev):
    """Markdown section: regional USGS gauges ranked by how unusual their peak was, then Flagstaff gauges."""
    from . import streams
    st = _streams(ev)
    if st is None:
        return '*Streamflow data could not be reached when this page was built.*'
    near, flag, cty = st
    out = [f'Peak flow during the event at USGS gauges within '
           f'{streams.NEAR_KM * 0.621:.0f} miles of Flagstaff with at least {streams.MIN_YEARS} years of annual '
           'peaks, ranked by where the peak falls in each gauge’s record. The typical yearly peak is the median '
           'annual peak. The last column compares the highest daily mean flow during the event with the 90th '
           'percentile of daily flow for that date (▲ = above it, i.e. well above normal for the time of year). '
           'Event flows are provisional real-time data and may be revised; the record is the published '
           'annual-peak series. Each gauge’s peak is taken from the start of the rain until the gauge has had time '
           'to crest: one day after the rain ends for basins under 100 sq mi, two days under 2,000 sq mi, three '
           'days under 6,000 sq mi (the Verde) and a week for the Little Colorado, but stopping once the next '
           f'storm’s runoff could arrive (a day with {streams.NEXT_RAIN:.2f} in or more at the airport, plus a '
           'third of the gauge’s crest time).\n']
    above = near.above & near.dmax.notna()
    if above.any():
        n = int((near.dmax.notna() & (near.dyears >= streams.MIN_YEARS)).sum())
        k = int(above.sum())
        out.append(f'{"All" if k == n else f"{k} of the"} {n} gauges with long daily records ran above the 90th percentile '
                   '(the table lists the most unusual peaks).\n')
    if len(near):
        out.append(streams.regional_table(near).to_markdown(index=False, disable_numparse=True))
    ft = streams.city_table(flag, cty)
    if len(ft):
        out.append('\n**Flagstaff.** USGS gauges within 20 miles of town, and the City of Flagstaff stream-stage '
                   'sensors that rose at least half a foot, with peaks taken through 6 hours after the rain (these '
                   'small basins crest within an hour or two). City sensors report water depth (ft), not flow, and '
                   'have records of only a few years. Their readings are screened for sensor glitches: a crest '
                   'counts only if the water passed through intermediate levels on the way up and down.\n')
        out.append(ft.to_markdown(index=False, disable_numparse=True))
    return '\n'.join(out)


def setup_text(slug):
    """Hand-written setup and impacts text for an event (site/_setup.yml), or {}."""
    import yaml
    if not SETUP_FILE.exists():
        return {}
    q = {str(k): v for k, v in (yaml.safe_load(SETUP_FILE.read_text(encoding='utf-8')) or {}).items()}
    return q.get(slug) or {}


def _paragraphs(lead, rain, cause, streams=()):
    paras = []
    if lead.get('setup'):
        paras.append('**How it came together.** ' + lead['setup'].strip())
        paras.append('**What fell.** ' + ' '.join(rain))
    else:
        paras.append(' '.join(rain))
    tail = ' '.join(cause)
    impacts = ' '.join(([lead['impacts'].strip()] if lead.get('impacts') else []) + list(streams))
    if lead.get('setup'):
        if tail:
            paras.append('**Why so much.** ' + tail)
        if impacts:
            paras.append('**What it did.** ' + impacts)
    elif tail or impacts:
        paras.append(' '.join(x for x in (tail, impacts) if x))
    return '\n\n'.join(paras)


def _cause_short(ev, st):
    """ERA5 diagnosis condensed for summaries that already have a written setup."""
    e = ev.era5_stats()
    out = []
    if e['ivt_max'] >= AR_IVT and st['frac_ar'] >= 0.5:
        s = (f'The atmospheric river drove the storm: {100 * st["frac_ar"]:.0f}% of the rain fell while it was '
             'overhead')
        s += (f', and at its peak it carried about {e["ivt_max"] / AR_IVT:.1f} times the moisture flow that defines '
              'one.' if e['ivt_max'] >= 2 * AR_IVT else '.')
        out.append(s)
    elif e['ivt_max'] >= AR_IVT:
        out.append(f'The flow of moisture reached atmospheric-river strength for part of the storm, but '
                   f'{100 * (1 - st["frac_ar"]):.0f}% of the rain fell outside those hours.')
    else:
        out.append('The flow of moisture stayed below atmospheric-river strength throughout.')
    for c in st['tc_connected']:
        out.append(f'Part of that moisture came directly from {_tc_name(c["type"], c["name"])}, '
                   f'about {_mi(c["km"])} miles to the {_dir_words(c["bearing"])}.')
    return out


def _cause_full(ev, st):
    out = []
    e = ev.era5_stats()
    cl = st['closed_low']
    steer = (f'an area of low pressure aloft centred about {_mi(cl["km"])} miles to the {_dir_words(cl["bearing"])}'
             if cl else 'a trough of low pressure to the west' if st['trough_lon'] is not None
             else 'the passing weather system')
    src = f'from the {_dir_words(e["ivt_dir"])}'
    if e['ivt_max'] >= AR_IVT and st['frac_ar'] >= 0.5:
        out.append(f'The storm was fed by an atmospheric river, a long, narrow stream of moist air flowing {src}, '
                   f'drawn north by {steer}. {100 * st["frac_ar"]:.0f}% of the rain fell while that stream '
                   f'was overhead.')
    elif e['ivt_max'] >= AR_IVT:
        out.append(f'Moist air flowed {src}, drawn north by {steer}. For part of the storm it was strong enough to '
                   f'count as an atmospheric river (a long, narrow stream of moist air), but most of the rain, '
                   f'{100 * (1 - st["frac_ar"]):.0f}%, fell outside those hours.')
    else:
        out.append(f'Moist air flowing {src} was lifted by {steer}, but the flow of moisture stayed below '
                   f'atmospheric-river strength.')
    if e['ivt_max'] >= 2 * AR_IVT:
        out.append(f'At its peak it carried about {e["ivt_max"] / AR_IVT:.1f} times the moisture flow that defines '
                   f'an atmospheric river.')
    for c in st['tc_connected']:
        out.append(f'Part of that moisture came directly from {_tc_name(c["type"], c["name"])}, '
                   f'about {_mi(c["km"])} miles to the {_dir_words(c["bearing"])}.')
    return out


def summary(ev):
    """Plain-language summary, then the technical summary in a collapsed callout."""
    return (plain_summary(ev) + '\n\n::: {.callout-note collapse="true"}\n## Technical summary\n\n'
            + narrative(ev) + '\n:::\n')


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


AFD_FILE = ROOT / 'site' / '_afd_quotes.yml'
SETUP_FILE = ROOT / 'site' / '_setup.yml'
AFD_URL = 'https://mesonet.agron.iastate.edu/wx/afos/p.php?pil=AFDFGZ&e={utc}'


def forecaster_notes(*slugs):
    """Markdown list of NWS Flagstaff Area Forecast Discussion excerpts for the given event slugs."""
    import yaml
    q = {str(k): v for k, v in (yaml.safe_load(AFD_FILE.read_text(encoding='utf-8')) or {}).items()}  # keys parse as dates
    items = sorted((it for s in slugs for it in q.get(s, [])), key=lambda it: it['utc'])
    if not items:
        return ''
    lines = ['The National Weather Service office in Flagstaff discussed the moisture source in its Area Forecast '
             'Discussions as the event unfolded. Excerpts, quoted verbatim (links to the Iowa Environmental '
             'Mesonet archive):\n']
    lines += [f'- **{it["local"]}** ([AFD]({AFD_URL.format(utc=it["utc"])})): "{it["text"]}"' for it in items]
    return '\n'.join(lines)
