"""City of Flagstaff ALERT rain-gauge network (https://rain.flagstaffaz.gov/jefmap/).

The tabular API returns one record per 1-mm bucket tip as a cumulative accumulator
(inches) that occasionally resets; increments are recovered by differencing.
"""
import json
import time
import urllib.request

import pandas as pd

from ..config import EXCLUDE_GAUGES

API = 'https://rain.flagstaffaz.gov/api'


def _retry(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def _get(path):
    req = urllib.request.Request(API + path, headers={'User-Agent': 'Flag-weather', 'Accept': 'application/json'})
    return _retry(lambda: json.loads(urllib.request.urlopen(req, timeout=120).read().decode()))


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
