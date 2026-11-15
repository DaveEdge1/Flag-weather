"""USGS streamflow: instantaneous discharge (NWIS water services) and annual peak-flow records.

Event peaks come from provisional real-time data and may be revised; annual peaks are the published record.
"""
import io
import json
import time
import urllib.request

import numpy as np
import pandas as pd

from ..config import DATA, FLAGSTAFF

DIR = DATA / 'usgs'
DIR.mkdir(exist_ok=True)
IV = 'https://waterservices.usgs.gov/nwis/iv/?format=json&bBox={bbox}&parameterCd=00060&startDT={start}&endDT={end}' \
     '&siteStatus=all'
PEAK = 'https://nwis.waterdata.usgs.gov/nwis/peak?site_no={site}&agency_cd=USGS&format=rdb'
SITE_URL = 'https://waterdata.usgs.gov/monitoring-location/USGS-{site}/'
BBOX = '-112.6,33.9,-109.8,36.0'     # Verde, upper Salt (Tonto), Little Colorado and Cataract basins around Flagstaff


def _read(url, tries=3):
    req = urllib.request.Request(url, headers={'User-Agent': 'Flag-weather'})
    for i in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=120).read().decode()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def _km(lat, lon):
    la0, lo0 = np.radians(FLAGSTAFF['lat']), np.radians(FLAGSTAFF['lon'])
    la, lo = np.radians(lat), np.radians(lon)
    a = np.sin((la - la0) / 2) ** 2 + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def event_peaks(start, end, bbox=BBOX):
    """Peak instantaneous discharge (cfs) at every gauge in bbox between start and end (dates, local)."""
    d = json.loads(_read(IV.format(bbox=bbox, start=start, end=end)))
    rows = []
    for ts in d['value']['timeSeries']:
        si = ts['sourceInfo']
        v = [(x['dateTime'], float(x['value'])) for x in ts['values'][0]['value'] if float(x['value']) > -999]
        if not v:
            continue
        t, q = max(v, key=lambda x: x[1])
        g = si['geoLocation']['geogLocation']
        rows.append(dict(site=si['siteCode'][0]['value'], name=si['siteName'], peak=q,
                         time=pd.Timestamp(t).tz_convert('America/Phoenix').tz_localize(None),
                         lat=g['latitude'], lon=g['longitude']))
    df = pd.DataFrame(rows).set_index('site')
    df['km'] = _km(df.lat, df.lon)
    return df


def annual_peaks(site):
    """Published annual peak discharge (cfs) by water year, cached."""
    f = DIR / f'peak_{site}.txt'
    if not f.exists() or time.time() - f.stat().st_mtime > 30 * 86400:
        f.write_text(_read(PEAK.format(site=site)), encoding='utf-8')
    lines = [ln for ln in f.read_text(encoding='utf-8').splitlines(True) if not ln.startswith('#')]
    if len(lines) < 3:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    pk = pd.read_csv(io.StringIO(''.join(lines[:1] + lines[2:])), sep='\t', dtype=str)
    pk['q'] = pd.to_numeric(pk.peak_va, errors='coerce')
    pk['dt'] = pd.to_datetime(pk.peak_dt.str.replace('-00', '-01'), format='%Y-%m-%d', errors='coerce')
    pk = pk.dropna(subset=['q', 'dt'])
    return pk.set_index('dt').q.sort_index()


def ranked(start, end, before):
    """Event peaks with their standing in the annual-peak record before `before` (the event's water year):
    rank, years of record, previous record and its date, and the ratio to the median annual peak."""
    ev = event_peaks(start, end)
    rows = []
    for site, r in ev.iterrows():
        pk = annual_peaks(site)
        pk = pk[pk.index < pd.Timestamp(before)]
        if len(pk) < 5 or r.peak <= 0:
            continue
        rows.append(dict(site=site, rank=int((pk > r.peak).sum() + 1), n=len(pk), first=int(pk.index.year.min()),
                         rec=pk.max(), rec_date=pk.idxmax(), median=pk.median(), ratio=r.peak / pk.median()))
    return ev.join(pd.DataFrame(rows).set_index('site'), how='inner')
