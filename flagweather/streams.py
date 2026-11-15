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


NEXT_RAIN = 0.20      # in; a day with this much at the airport starts a new storm and ends the crest windows
CITY_LAG = pd.Timedelta('6h')   # city sensors (basins of a few sq mi) crest within hours of the rain


def _lag(area):
    """How long after the rain a gauge may still be cresting, from its drainage area (sq mi): small creeks
    within a day, mid-sized ones (Oak Creek, Chevelon) in two, the Verde in about three, and the Little
    Colorado up to a week, depending on where in its basin the rain fell."""
    if pd.isna(area):
        return pd.Timedelta('3D')
    if area < 100:
        return pd.Timedelta('1D')
    if area < 2000:
        return pd.Timedelta('2D')
    if area < 6000:
        return pd.Timedelta('3D')
    return pd.Timedelta('7D')


def _local(t):
    return t - pd.Timedelta('7h')


def _next_rain(ev):
    """Local midnight of the first day after the event with NEXT_RAIN at the airport (or None)."""
    rec = getattr(ev, 'precip_record', None)
    if rec is None:
        from .sources import acis
        rec = acis.precip_record()
    last_day = _local(ev.end).normalize()
    after = rec[(rec.index > last_day) & (rec.index <= last_day + pd.Timedelta('8D'))]
    wet = after[after >= NEXT_RAIN]
    return wet.index[0] if len(wet) else None


def _ends(ev):
    """Function site -> end of that gauge's crest window (local), and the end for the whole request (None)."""
    areas = usgs.drainage_areas()
    end, cap = _local(ev.end), _next_rain(ev)

    def ends(site):
        lag = pd.Timedelta('7D') if site is None else _lag(areas.get(site))
        # runoff from the next storm also takes time to reach a gauge: a third of its crest lag
        return min(end + lag, cap + lag / 3) if cap is not None else end + lag
    ends.cap = cap
    return ends


def nice_name(name):
    n = name.replace(', AZ.', '').replace(', AZ', '').replace(', ARIZ.', '').title()
    for a, b in {' Nr ': ' near ', ' Abv ': ' above ', ' Blw ': ' below ', ' At ': ' at ', ' Near ': ' near ',
                 ' Above ': ' above ', ' Below ': ' below ', 'Rvr': 'River', ' R ': ' River ', ' Cr ': ' Creek '}.items():
        n = n.replace(a, b)
    return n


# ---- regional USGS gauges ----------------------------------------------------------------------
def regional(ev):
    a, ends = _local(ev.start), _ends(ev)
    df = usgs.ranked(a, ends, _water_year_start(ev.start))
    df['name'] = df.name.map(nice_name)
    df['standing'] = df['rank'] / (df.n + 1)
    trimmed = lambda site: ends(None) if site is None else df.window_end.get(site, ends(site))  # noqa: E731
    vn = usgs.vs_normal(a, trimmed, df.index[(df.km <= NEAR_KM)])
    df = df.join(vn) if len(vn) else df.assign(dmax=np.nan, p50=np.nan, p90=np.nan, p95=np.nan)
    df['above'] = (df.dmax > df.p90) & (df.dmax >= 1) & (df.dyears >= MIN_YEARS)
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


def _flat_high(w, level):
    """True if the sensor held a reading at or above `level` unchanged (within 0.02 ft) for 6 h or more:
    a stuck sensor, since water draining a wash never holds that steady."""
    high = (w >= level) & (w.shift() >= level)            # consecutive readings, both high
    gap = w.index.to_series().diff() >= pd.Timedelta('6h')
    return bool((high & gap & (w.diff().abs() <= 0.02)).any())


def credible_peak(s, floor=-np.inf, within=None, dry=None):
    """Highest crest in s (raw readings, local-time index) that behaves like water rather than a sensor glitch:
    the stage passes through intermediate levels (at least 2 readings between 25% and 90% of the crest's height
    above the lowest reading within 24 h) in the 12 h before the crest and in the 24 h after it, and is below
    half that height sometime in the 24 h before and the 48 h after, and never holds steady above half that
    height for 6 h (both rule out sensors stuck high).
    The crest must also rise from near the sensor's usual dry level (the median reading): the lowest reading
    within 24 h is no more than half the rise above it (rules out sensors drifting at a false high level).
    Failed crests are discarded with the readings around them, and the next-highest is tried.
    Returns (value, time) or (nan, None)."""
    s = s.dropna()
    dry = s.median() if dry is None else dry
    for _ in range(300):
        s = s[s > floor]
        c = s if within is None else s[within[0]:within[1]]
        if not len(c):
            break
        t = c.idxmax()
        v = s[t]
        base = s[t - pd.Timedelta('24h'):t + pd.Timedelta('24h')].min()
        rise = v - base
        lo, hi, mid = base + 0.25 * rise, base + 0.9 * rise, base + 0.5 * rise
        up = s[t - pd.Timedelta('12h'):t].iloc[:-1]
        down = s[t:t + pd.Timedelta('24h')].iloc[1:]
        ok = (rise > 0 and base <= dry + 0.5 * rise and up.between(lo, hi).sum() >= 2 and down.between(lo, hi).sum() >= 2
              and (s[t - pd.Timedelta('24h'):t] < mid).any() and (s[t:t + pd.Timedelta('48h')] < mid).any()
              and not _flat_high(s[t - pd.Timedelta('48h'):t + pd.Timedelta('48h')], mid))
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
    a, b = _local(ev.start), _local(ev.end) + CITY_LAG
    cap = _next_rain(ev)
    if cap is not None:
        b = min(b, cap)
    hist = _history()
    sites = gauges.stream_sites()
    rows = []
    for k, r in sites.iterrows():
        s = gauges._stage(k, f'{(a - pd.Timedelta("1D")):%Y-%m-%d}', f'{(b + pd.Timedelta("2D")):%Y-%m-%d}')
        if not len(s):
            continue
        s.index = s.index.tz_convert('America/Phoenix').tz_localize(None)
        s = s.sort_index()[a - pd.Timedelta('1D'):b + pd.Timedelta('2D')]   # context for the glitch screen
        if not len(s[a:b]):
            continue
        peak, t = credible_peak(s, within=(a, b))
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
            v, tt = credible_peak(h[hh.index[-1] - pd.Timedelta('3D'):hh.index[-1] + pd.Timedelta('3D')], dry=h.median())
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


def _count(k, n):
    if k == n:
        return f'all {n}'
    return f'{k} of the {n}' if k < 10 and n < 10 else f'{k} of {n}'


def _join(items):
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


def _short(name):
    """'Chevelon Fork below Wildcat Canyon, near Winslow' -> 'Chevelon Fork near Winslow'."""
    head, _, tail = name.partition(',')
    if tail.strip().startswith('near'):
        head = head.split(' below ')[0].split(' above ')[0] + ' ' + tail.strip()
    return ('the ' if ' River ' in head + ' ' else '') + head


MIN_RATIO = 0.5       # gauges named for running high must peak at >= this fraction of their median annual peak
MIN_CITY_RISE = 1.0   # ft; city sensors named in the summary rose at least this much


def _typical(x):
    if x >= 1.3:
        return f'about {x:.0f}' if x >= 2.95 else f'{x:.1f}'
    return ''


def _vs_typical(r):
    x = r.ratio
    if x >= 1.3:
        return f'{_typical(x)} times its typical yearly peak'
    if x >= 0.8:
        return 'about its typical yearly peak'
    return f'about {round(x * 10) * 10:.0f}% of its typical yearly peak'


def sentences(near, flag, cty, start=None, n_regional=4, n_high=3, n_city=2):
    """Plain-language sentences on stream peaks (cfs written out on first use): how many regional gauges ran
    above the 90th percentile of daily flow for the date; gauges at or near their records (top 3 annual peaks);
    otherwise the gauges that ran highest against their typical yearly peak; Flagstaff gauges near their records
    and city sensors at their highest in more than a year. Empty when no gauge ran well above normal."""
    out, unit = [], [' cubic feet per second (cfs)']

    def q(x, in_parens=False):
        u = unit.pop() if unit else ' cfs'
        if in_parens:
            u = u.replace(' (cfs)', ', or cfs')
        return f'{x:,.0f}{u}'

    has_daily = near.dmax.notna() & (near.dyears >= MIN_YEARS)
    high = near[near.above & (near.ratio >= MIN_RATIO)].sort_values('ratio', ascending=False)
    top = near[near['rank'] <= 3].head(n_regional)
    flag = flag[flag['rank'] <= 3]
    if len(cty):
        old = cty.last_higher.isna() | (cty.last_higher < pd.Timestamp(start) - pd.Timedelta('365D'))
        cty = cty[old & (cty.rise >= MIN_CITY_RISE)]
    if not (len(high) or len(top) or len(flag) or len(cty)):
        return []
    if near.above.any():
        out.append(f'{_cap(_count(near.above.sum(), has_daily.sum()))} USGS gauges within {NEAR_KM * 0.621:.0f} '
                   'miles had daily flows above the 90th percentile for the date.')
    rec = top[top['rank'] == 1]
    for _, r in rec.iterrows():
        out.append(f'{_cap(_short(r["name"]))} set a record of {q(r.peak)}, topping {_cfs(r.rec)} in '
                   f'{r.rec_date:%B %Y} ({r.n} years of USGS measurements).')
    for k in (2, 3):
        g = top[top['rank'] == k]
        if len(g):
            items = [f'{_short(r["name"])} ({q(r.peak, True)}, behind {r.rec_date:%B %Y})' for _, r in g.iterrows()]
            out.append(f'{_cap(_join(items))} had {"its" if len(g) == 1 else "their"} {_ord(k)}-highest '
                       f'peak{"s" * (len(g) > 1)} on record.')
    if not len(top):
        groups = {}           # gauges sharing the same comparison are named together
        for _, r in high.head(n_high).iterrows():
            groups.setdefault(_vs_typical(r), []).append(r)
        items = []
        for phrase, rs in groups.items():
            if len(rs) == 1:
                items.append(f'{_short(rs[0]["name"])} peaked at {q(rs[0].peak)}, {phrase}')
            else:
                names = _join([f'{_short(r["name"])} ({q(r.peak, True)})' for r in rs])
                items.append(f'{names} {"both" if len(rs) == 2 else "all"} peaked at {phrase.replace("its typical yearly peak", "their typical yearly peaks")}')
        if items:
            items[0] += ' (the median of annual peaks)'
            out.append(_cap('; '.join(items)) + '.')
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


def _daily(r):
    if pd.isna(r.get('dmax')) or not r.get('dyears', 0) >= MIN_YEARS or pd.isna(r.p90):
        return '–'
    return f'{r.dmax:,.0f} vs {r.p90:,.{0 if r.p90 >= 10 else 1 if r.p90 >= 1 else 2}f}' + (' ▲' if r.above else '')


def regional_table(near, n=8):
    rows = [{'Gauge': f'[{r["name"]}]({usgs.SITE_URL.format(site=s)})',
             'Peak (cfs)': f'{r.peak:,.0f}', 'Time (MST)': f'{r.time:%b} {r.time.day}, {r.time:%H:%M}',
             'Rank': f'{r["rank"]} of {r.n} (since {r["first"]})' if r['rank'] <= r.n
                     else f'below all {r.n} (since {r["first"]})',
             'Previous record (cfs)': f'{r.rec:,.0f} ({r.rec_date:%b %Y})',
             'Typical yearly peak (cfs)': f'{r["median"]:,.0f}',
             'Daily flow vs 90th pct (cfs)': _daily(r)}
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
