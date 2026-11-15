"""City of Flagstaff ALERT rain-gauge network (https://rain.flagstaffaz.gov/jefmap/).

The tabular API returns one record per 1-mm bucket tip as a cumulative accumulator
(inches) that occasionally resets; increments are recovered by differencing.
"""
import json
import time
import urllib.error
import urllib.request

import pandas as pd

from ..config import DATA, EXCLUDE_GAUGES

API = 'https://rain.flagstaffaz.gov/api'


def _retry(fn, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1 or (isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500):
                raise
            time.sleep(5 * (i + 1))


def _get(path):
    req = urllib.request.Request(API + path, headers={'User-Agent': 'Flag-weather', 'Accept': 'application/json'})
    return _retry(lambda: json.loads(urllib.request.urlopen(req, timeout=45).read().decode()))


def network():
    """Flagstaff-area rain gauges: id, site, owner, lat/lon and latest rolling totals (in)."""
    d = _get('/precip.json')['data']
    rows = [dict(id=k, site=v['site'], owner=v['owner'], lat=v['latitude'], lon=v['longitude'],
                 latest=v['latest'], **{f'last_{p}': v['depths'].get(p) for p in ['P1D', 'P7D', 'P30D']})
            for k, v in d.items() if k not in EXCLUDE_GAUGES]
    return pd.DataFrame(rows).set_index('id')


def tips(gauge_id, start, end):
    """Rain increments (in) at each bucket tip; index is tz-aware local time."""
    d = _get(f'/tabular/{gauge_id}?startDate={start}&endDate={end}')['data']
    if not d:
        return pd.Series(dtype=float, name=gauge_id)
    s = pd.Series({pd.Timestamp(r['ts']): r['precipitation'] for r in d}).sort_index()
    s = s[~s.index.duplicated()]
    inc = s.diff()
    inc[(inc < 0) | (inc > 1.0)] = 0.0   # accumulator resets / spikes
    return inc.fillna(0.0).rename(gauge_id)


def hourly(gauge_ids, start, end):
    """Hourly totals (in), UTC index (naive), one column per gauge."""
    cols = {}
    for g in gauge_ids:
        s = tips(g, start, end)
        if len(s):
            s.index = s.index.tz_convert('UTC').tz_localize(None)
            cols[g] = s.resample('1h').sum()
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta('1D'), freq='1h', inclusive='left')
    return pd.DataFrame(cols).reindex(idx).fillna(0.0)


# ---- stream-stage sensors ----------------------------------------------------------------------
STAGE_DIR = DATA / 'stage'
STAGE_DIR.mkdir(exist_ok=True)
# Sensors in detention basins and ponds (not stream channels) are left out of stream comparisons
NOT_CHANNELS = {'alert1/3333', 'alert1/3393', 'alert1/3403', 'alert1/3373'}


def stream_sites():
    """City of Flagstaff stream-stage sensors: id, site, lat/lon."""
    d = _get('/stream.json')['data']
    rows = [dict(id=k, site=v['site'], owner=v['owner'], lat=v['latitude'], lon=v['longitude'])
            for k, v in d.items() if v['owner'] == 'Flagstaff' and k not in NOT_CHANNELS]
    return pd.DataFrame(rows).set_index('id')


def _stage(gauge_id, start, end):
    try:
        d = _get(f'/tabular/{gauge_id}?startDate={start}&endDate={end}')['data']
    except urllib.error.HTTPError as e:   # too many readings for one request: split the range
        a, b = pd.Timestamp(start), pd.Timestamp(end)
        if e.code != 413 or b - a < pd.Timedelta('2D'):
            raise
        m = a + (b - a) / 2
        return pd.concat([_stage(gauge_id, f'{a:%Y-%m-%d}', f'{m:%Y-%m-%d}'),
                          _stage(gauge_id, f'{m + pd.Timedelta("1D"):%Y-%m-%d}', f'{b:%Y-%m-%d}')]).sort_index()
    s = pd.Series({pd.to_datetime(r['ts'], utc=True): r['stage'] for r in d if r.get('stage') is not None},
                  dtype=float)
    return s.sort_index()


def stage_history(gauge_id, through):
    """Stage (ft) since the sensor's first record (2019 onward), cached by quarter; UTC index."""
    f = STAGE_DIR / f'{gauge_id.replace("/", "_")}.csv'
    old = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0] if f.exists() else pd.Series(dtype=float)
    first = (old.index.max().tz_convert(None).normalize() - pd.Timedelta('1D')) if len(old) \
        else pd.Timestamp('2019-01-01')
    parts = [old]
    for a in pd.date_range(first.to_period('Q').start_time, through, freq='QS'):
        b = min(a + pd.offsets.QuarterEnd(0), pd.Timestamp(through))
        parts.append(_stage(gauge_id, f'{max(a, first):%Y-%m-%d}', f'{b:%Y-%m-%d}'))
    s = pd.concat([p for p in parts if len(p)])
    s = s[~s.index.duplicated(keep='last')].sort_index()
    s.rename('stage').to_csv(f)
    return s
