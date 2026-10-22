"""NWS station data via RCC-ACIS (Flagstaff Pulliam Airport threaded record)."""
import json
import time
import urllib.request

import pandas as pd

from ..config import ACIS_SID, DATA

API = 'https://data.rcc-acis.org'


def _retry(fn, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def _post(path, q):
    req = urllib.request.Request(API + path, data=json.dumps(q).encode(),
                                 headers={'Content-Type': 'application/json', 'User-Agent': 'Flag-weather'})
    return _retry(lambda: json.loads(urllib.request.urlopen(req, timeout=45).read().decode()))


def _num(v):
    if v in ('M', 'S', '', None):
        return float('nan')
    if v == 'T':
        return 0.0
    return float(str(v).rstrip('AS'))


def daily(start, end, sid=ACIS_SID):
    """Daily max/min/mean temperature (F), precipitation, snowfall, snow depth (in)."""
    elems = ['maxt', 'mint', 'avgt', 'pcpn', 'snow', 'snwd']
    d = _post('/StnData', {'sid': sid, 'sdate': str(start), 'edate': str(end), 'elems': elems})['data']
    df = pd.DataFrame(d, columns=['date'] + elems).set_index('date')
    df.index = pd.to_datetime(df.index)
    return df.apply(lambda c: c.map(_num))


def daily_normals(start, end, sid=ACIS_SID):
    """1991-2020 daily normals: high, low (F), precipitation, snowfall (in)."""
    elems = [{'name': n, 'normal': '1'} for n in ('maxt', 'mint', 'pcpn', 'snow')]
    d = _post('/StnData', {'sid': sid, 'sdate': str(start), 'edate': str(end), 'elems': elems})['data']
    df = pd.DataFrame(d, columns=['date', 'maxt_n', 'mint_n', 'pcpn_n', 'snow_n']).set_index('date')
    df.index = pd.to_datetime(df.index)
    return df.apply(lambda c: c.map(_num))


def water_year_to_date(date, sid=ACIS_SID):
    """Observed and normal precipitation (in) from Oct 1 through date."""
    e = {'name': 'pcpn', 'interval': 'dly', 'duration': 'std', 'season_start': '10-01', 'reduce': 'sum'}
    d = _post('/StnData', {'sid': sid, 'sdate': str(date), 'edate': str(date), 'elems': [e, dict(e, normal='1')]})
    return _num(d['data'][0][1]), _num(d['data'][0][2])


def precip_record(sid=ACIS_SID, refresh_days=1):
    """Full daily precipitation record (in), cached for a day."""
    f = DATA / f'acis_{sid}_pcpn.csv'
    if f.exists() and (time.time() - f.stat().st_mtime) < refresh_days * 86400:
        return pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
    d = _post('/StnData', {'sid': sid, 'sdate': 'por', 'edate': 'por', 'elems': ['pcpn']})['data']
    s = pd.Series({pd.Timestamp(a): _num(b) for a, b in d}, name='pcpn')
    s.to_csv(f)
    return s


def rank_in_month(total, ndays, month, record):
    """Rank of an n-day total among each year's largest n-day total ending in the same calendar month."""
    roll = record.rolling(ndays, min_periods=ndays).sum()
    annual = roll[roll.index.month == month].groupby(roll[roll.index.month == month].index.year).max().dropna()
    return int((annual > total + 1e-9).sum() + 1), len(annual), int(annual.index.min())
