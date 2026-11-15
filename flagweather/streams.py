"""Streamflow during an event: regional USGS gauges ranked by how unusual their peak was, and City of Flagstaff
stream-stage sensors compared with their own (short) records."""
import numpy as np
import pandas as pd

from .config import DERIVED
from .sources import gauges, usgs

STAGE_RAW = DERIVED / 'stage_raw.csv.gz'      # committed: every city stream-stage reading (ft), all sensors
NEAR_KM = 110          # regional gauges considered "nearby"
MIN_YEARS = 20         # regional gauges need this many annual peaks to be ranked
FLAG_KM = 30           # USGS gauges this close count as Flagstaff gauges, whatever their record length
MIN_STAGE_RISE = 0.5   # ft; city sensors that rose less than this are not reported


def _water_year_start(t):
    t = pd.Timestamp(t)
    return pd.Timestamp(f'{t.year if t.month >= 10 else t.year - 1}-10-01')


def _window(ev):
    """Local dates covering the event plus two days for rivers to crest."""
    a = (ev.start - pd.Timedelta('7h')).normalize()
    b = (ev.end - pd.Timedelta('7h')).normalize() + pd.Timedelta('2D')
    return a, b


def nice_name(name):
    n = name.replace(', AZ.', '').replace(', AZ', '').replace(', ARIZ.', '').title()
    for a, b in {' Nr ': ' near ', ' Abv ': ' above ', ' Blw ': ' below ', ' At ': ' at ', ' Near ': ' near ',
                 ' Above ': ' above ', ' Below ': ' below ', 'Rvr': 'River', ' R ': ' River ', ' Cr ': ' Creek '}.items():
        n = n.replace(a, b)
    return n


# ---- regional USGS gauges ----------------------------------------------------------------------
def regional(ev):
    a, b = _window(ev)
    df = usgs.ranked(f'{a:%Y-%m-%d}', f'{b:%Y-%m-%d}', _water_year_start(ev.start))
    df['name'] = df.name.map(nice_name)
    df['standing'] = df['rank'] / (df.n + 1)
    near = df[(df.km <= NEAR_KM) & (df.n >= MIN_YEARS)].sort_values(['standing', 'ratio'], ascending=[True, False])
    flag = df[df.km <= FLAG_KM].sort_values('km')
    return near, flag


# ---- city stage sensors --------------------------------------------------------------------------
def build_stage_raw(through):
    """Rebuild the committed file of all city stage readings from the cached sensor histories (run locally)."""
    sites = gauges.stream_sites()
    parts = []
    for k in sites.index:
        s = gauges.stage_history(k, through)
        if len(s):
            parts.append(pd.DataFrame({'id': k, 'time': s.index.tz_convert(None), 'stage': s.values}))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(STAGE_RAW, index=False, float_format='%.2f', date_format='%Y-%m-%dT%H:%M:%S')
    return df


def credible_peak(s, floor=-np.inf):
    """Highest crest in s (raw readings, local-time index) that behaves like water rather than a sensor glitch:
    the stage passes through intermediate levels (at least 2 readings between 25% and 90% of the crest's height
    above the lowest reading within 24 h) in the 12 h before the crest and in the 24 h after it, and is below
    half that height sometime in the 24 h before and the 48 h after (which rules out sensors stuck high).
    Failed crests are discarded with the readings around them, and the next-highest is tried.
    Returns (value, time) or (nan, None)."""
    s = s.dropna()
    for _ in range(300):
        s = s[s > floor]
        if not len(s):
            break
        t = s.idxmax()
        v = s[t]
        base = s[t - pd.Timedelta('24h'):t + pd.Timedelta('24h')].min()
        rise = v - base
        lo, hi, mid = base + 0.25 * rise, base + 0.9 * rise, base + 0.5 * rise
        up = s[t - pd.Timedelta('12h'):t].iloc[:-1]
        down = s[t:t + pd.Timedelta('24h')].iloc[1:]
        ok = (rise > 0 and up.between(lo, hi).sum() >= 2 and down.between(lo, hi).sum() >= 2
              and (s[t - pd.Timedelta('24h'):t] < mid).any() and (s[t:t + pd.Timedelta('48h')] < mid).any())
        i = s.index.get_loc(t)
        nb = s.iloc[max(i - 1, 0):i + 2].drop(t)
        supported = len(nb) and nb.max() >= base + 0.75 * rise
        if ok and supported:
            return v, t
        if ok:              # a lone spike on top of a real crest: drop just that reading
            s = s.drop(t)
        else:
            w = s[t - pd.Timedelta('3h'):t + pd.Timedelta('3h')]
            s = s.drop(w[w >= mid].index)
    return np.nan, None


def _history():
    df = pd.read_csv(STAGE_RAW, parse_dates=['time'])
    df['time'] = df.time.dt.tz_localize('UTC').dt.tz_convert('America/Phoenix').dt.tz_localize(None)
    return {k: g.set_index('time').stage.sort_index() for k, g in df.groupby('id')}


def city(ev):
    """Peak stage (glitch-screened; see credible_peak) at each city channel sensor during the event, against the sensor's earlier record."""
    a, b = _window(ev)
    hist = _history()
    sites = gauges.stream_sites()
    rows = []
    for k, r in sites.iterrows():
        s = gauges._stage(k, f'{(a - pd.Timedelta("1D")):%Y-%m-%d}', f'{b:%Y-%m-%d}')
        if not len(s):
            continue
        s.index = s.index.tz_convert('America/Phoenix').tz_localize(None)
        s = s.sort_index()[a:b + pd.Timedelta('1D')]
        if not len(s):
            continue
        peak, t = credible_peak(s)
        if t is None:
            continue
        base = s[:t].min()
        h = hist.get(k, pd.Series(dtype=float))
        h = h[h.index < a]
        if peak - base < MIN_STAGE_RISE or len(h) == 0 or h.index[-1] - h.index[0] < pd.Timedelta('365D'):
            continue
        prev_max, prev_t = credible_peak(h)
        if prev_t is None:
            continue
        # most recent earlier credible crest at or above this one
        last = None
        hh = h[h >= peak]
        while len(hh):
            v, tt = credible_peak(h[hh.index[-1] - pd.Timedelta('3D'):hh.index[-1] + pd.Timedelta('3D')])
            if tt is not None and v >= peak:
                last = tt
                break
            hh = hh[hh.index < hh.index[-1] - pd.Timedelta('3D')]
        rows.append(dict(id=k, name=r.site.replace(' nr ', ' near ').replace('Ave.', 'Ave'), peak=peak, time=t, rise=peak - base, since=h.index.min(),
                         prev_max=prev_max, prev_max_date=prev_t, last_higher=last,
                         rel=peak / prev_max if prev_max > 0 else np.nan))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index('id').sort_values('rel', ascending=False)


# ---- text and tables -----------------------------------------------------------------------------
def _cfs(q):
    return f'{q:,.0f} cfs'


def _standing(r):
    if r['rank'] == 1:
        return f'highest in {r.n} years of record (previous record {_cfs(r.rec)}, {r.rec_date:%B %Y})'
    if r['rank'] == 2:
        return f'second-highest in {r.n} years of record, behind {r.rec_date:%B %Y} ({_cfs(r.rec)})'
    return f'{_ord(r["rank"])}-highest in {r.n} years of record'


def _times(x):
    return {2: 'twice', 3: 'three times', 4: 'four times'}.get(round(x), f'{x:.0f} times')


def _ord(n):
    return {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth'}.get(n, f'{n}th')


def _city_standing(r):
    yrs = f'since the sensor began in {r.since:%Y}'
    if pd.isna(r.last_higher):
        return f'its highest {yrs}'
    return f'its highest since {r.last_higher:%B %Y}'


def _cap(t):
    return t[0].upper() + t[1:]


def _join(items):
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


def _short(name):
    """'Chevelon Fork below Wildcat Canyon, near Winslow' -> 'Chevelon Fork near Winslow'."""
    head, _, tail = name.partition(',')
    if tail.strip().startswith('near'):
        head = head.split(' below ')[0].split(' above ')[0] + ' ' + tail.strip()
    return ('the ' if ' River ' in head + ' ' else '') + head


def sentences(near, flag, cty, n_regional=4, n_city=2):
    """Plain-language sentences on the most unusual stream peaks (cfs written out on first use)."""
    out, unit = [], [' cubic feet per second (cfs)']

    def q(x):
        u = unit.pop() if unit else ' cfs'
        return f'{x:,.0f}{u}'

    top = near.head(n_regional)
    rec = top[top['rank'] == 1]
    for _, r in rec.iterrows():
        out.append(f'{_cap(_short(r["name"]))} set a record of {q(r.peak)}, topping {_cfs(r.rec)} in '
                   f'{r.rec_date:%B %Y} ({r.n} years of USGS measurements).')
    for k in (2, 3):
        g = top[top['rank'] == k]
        if len(g):
            items = [f'{_short(r["name"])} ({q(r.peak)}, behind {r.rec_date:%B %Y})' for _, r in g.iterrows()]
            out.append(f'{_cap(_join(items))} had {"its" if len(g) == 1 else "their"} {_ord(k)}-highest '
                       f'peak{"s" * (len(g) > 1)} on record.')
    loc = []
    for _, r in flag.iterrows():
        if r['rank'] == 1 and r.peak >= 1.5 * r.rec:
            loc.append(f'{r["name"]} ran at {q(r.peak)}, about {_times(r.peak / r.rec)} its previous record '
                       f'({r.n} years of measurements)')
        elif r['rank'] <= 3:
            loc.append(f'{r["name"]} ran at {q(r.peak)}, the {_standing(r)}')
    for _, r in cty.head(n_city).iterrows():
        loc.append(f'{r["name"]} rose {r.rise:.1f} ft, {_city_standing(r)}')
    if loc:
        out.append('Closer to town, ' + '; '.join(loc) + '.')
    if out:
        out[-1] += ' (River flows are provisional.)'
    return out


def regional_table(near, n=6):
    rows = [{'Gauge': f'[{r["name"]}]({usgs.SITE_URL.format(site=s)})',
             'Distance (mi)': f'{r.km * 0.621:.0f}',
             'Peak (cfs)': f'{r.peak:,.0f}', 'Time (MST)': f'{r.time:%b} {r.time.day}, {r.time:%H:%M}',
             'Rank': f'{r["rank"]} of {r.n} (since {r["first"]})',
             'Previous record (cfs)': f'{r.rec:,.0f} ({r.rec_date:%b %Y})'}
            for s, r in near.head(n).iterrows()]
    return pd.DataFrame(rows)


def city_table(flag, cty):
    rows = [{'Gauge': f'[{r["name"]}]({usgs.SITE_URL.format(site=s)}) (USGS)', 'Peak': _cfs(r.peak),
             'Time (MST)': f'{r.time:%b} {r.time.day}, {r.time:%H:%M}',
             'Compared with its record': f'{r["rank"]} of {r.n} annual peaks (since {r["first"]}); previous record '
                                         f'{_cfs(r.rec)} ({r.rec_date:%b %Y})'}
            for s, r in flag.iterrows() if r.peak > 0]
    rows += [{'Gauge': f'{r["name"]} (city stage sensor)', 'Peak': f'{r.peak:.2f} ft (rose {r.rise:.1f} ft)',
              'Time (MST)': f'{r.time:%b} {r.time.day}, {r.time:%H:%M}',
              'Compared with its record': f'{_city_standing(r)[0].upper()}{_city_standing(r)[1:]}; previous '
                                          f'high {r.prev_max:.2f} ft ({r.prev_max_date:%b %Y})'}
             for _, r in cty.iterrows()]
    return pd.DataFrame(rows)
