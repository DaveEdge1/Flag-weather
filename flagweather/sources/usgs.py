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


SITES = 'https://waterservices.usgs.gov/nwis/site/?format=rdb&bBox={bbox}&siteOutput=expanded&siteType=ST'         '&hasDataTypeCd=iv&parameterCd=00060'


def drainage_areas(bbox=BBOX):
    """Drainage area (sq mi) of each stream gauge in bbox, cached for 30 days."""
    f = DIR / 'sites.rdb'
    if not f.exists() or time.time() - f.stat().st_mtime > 30 * 86400:
        f.write_text(_read(SITES.format(bbox=bbox)), encoding='utf-8')
    lines = [ln for ln in f.read_text(encoding='utf-8').splitlines(True) if not ln.startswith('#')]
    df = pd.read_csv(io.StringIO(''.join(lines[:1] + lines[2:])), sep='	', dtype=str)
    return pd.to_numeric(df.set_index('site_no').drain_area_va, errors='coerce')


def instantaneous(start, end, bbox=BBOX):
    """Instantaneous discharge (cfs; provisional) at every gauge in bbox between start and end (dates, local):
    {site: series with local-time index}, plus a DataFrame of name, lat, lon, km from Flagstaff."""
    d = json.loads(_read(IV.format(bbox=bbox, start=start, end=end)))
    series, meta = {}, []
    for ts in d['value']['timeSeries']:
        si = ts['sourceInfo']
        v = {pd.Timestamp(x['dateTime']): float(x['value']) for x in ts['values'][0]['value']
             if float(x['value']) > -999}
        if not v:
            continue
        site = si['siteCode'][0]['value']
        s = pd.Series(v)
        s.index = pd.DatetimeIndex(s.index).tz_convert('America/Phoenix').tz_localize(None)
        series[site] = s.sort_index()
        g = si['geoLocation']['geogLocation']
        meta.append(dict(site=site, name=si['siteName'], lat=g['latitude'], lon=g['longitude']))
    meta = pd.DataFrame(meta).set_index('site')
    meta['km'] = _km(meta.lat, meta.lon)
    return series, meta


def event_peaks(start, ends, bbox=BBOX):
    """Peak instantaneous discharge (cfs) at every gauge in bbox from start (local) to that gauge's end time
    (ends: function site -> local Timestamp), with its time, the window end and the distance from Flagstaff."""
    last = max(ends(None), start) + pd.Timedelta('1D')
    series, meta = instantaneous(f'{start:%Y-%m-%d}', f'{last:%Y-%m-%d}', bbox)
    rows = []
    cap = getattr(ends, 'cap', None)
    for site, s in series.items():
        end = ends(site)
        if cap is not None and end > cap:
            # after the next storm starts, stop at the low point before the river rises again
            tail = s[cap:end]
            if len(tail) and tail[tail.idxmin():].max() > 1.25 * max(tail.min(), 1):
                end = tail.idxmin()
        w = s[start:end]
        if w.empty:
            continue
        rows.append(dict(site=site, peak=w.max(), time=w.idxmax(), window_end=end))
    return meta.join(pd.DataFrame(rows).set_index('site'), how='inner')


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


DV = 'https://waterservices.usgs.gov/nwis/dv/?format=json&bBox={bbox}&parameterCd=00060&statCd=00003' \
     '&startDT={start}&endDT={end}&siteStatus=all'
STAT = 'https://waterservices.usgs.gov/nwis/stat/?format=rdb&sites={site}&statReportType=daily&statTypeCd=all' \
       '&parameterCd=00060'
PCTS = ['p05', 'p10', 'p20', 'p25', 'p50', 'p75', 'p80', 'p90', 'p95']


def daily_means(start, end, bbox=BBOX):
    """Daily mean discharge (cfs; provisional) at every gauge in bbox, as a DataFrame (date x site)."""
    d = json.loads(_read(DV.format(bbox=bbox, start=start, end=end)))
    cols = {}
    for ts in d['value']['timeSeries']:
        v = {pd.Timestamp(x['dateTime'][:10]): float(x['value']) for x in ts['values'][0]['value']
             if float(x['value']) > -999}
        if v:
            cols[ts['sourceInfo']['siteCode'][0]['value']] = pd.Series(v)
    return pd.DataFrame(cols)


def daily_stats(site):
    """Day-of-year statistics of daily mean discharge (approved record), cached: index (month, day);
    columns count, max, max_yr and the percentiles in PCTS."""
    f = DIR / f'stat_{site}.txt'
    if not f.exists() or time.time() - f.stat().st_mtime > 30 * 86400:
        try:
            f.write_text(_read(STAT.format(site=site)), encoding='utf-8')
        except Exception:     # the service answers 404 for gauges without statistics
            f.write_text('', encoding='utf-8')
    lines = [ln for ln in f.read_text(encoding='utf-8').splitlines(True) if not ln.startswith('#')]
    if len(lines) < 3:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(''.join(lines[:1] + lines[2:])), sep='\t', dtype=str)
    df = df[df.parameter_cd == '00060']
    out = pd.DataFrame({'count': pd.to_numeric(df.count_nu), 'max': pd.to_numeric(df.max_va, errors='coerce'),
                        'max_yr': pd.to_numeric(df.max_va_yr, errors='coerce')})
    for p in PCTS:
        out[p] = pd.to_numeric(df[f'{p}_va'], errors='coerce')
    out.index = pd.MultiIndex.from_arrays([df.month_nu.astype(int), df.day_nu.astype(int)])
    return out[~out.index.duplicated()]    # one time series per gauge (the first listed)


def vs_normal(start, ends, sites):
    """Highest daily mean flow from start to each site's end (local dates) against the statistics for that day of
    the year: dmax, its date, the median (p50), 90th and 95th percentiles for the date, years of daily record,
    and the highest daily mean for the date with its year."""
    dm = daily_means(f'{start:%Y-%m-%d}', f'{max(ends(None), start):%Y-%m-%d}')
    rows = []
    for site in sites:
        if site not in dm or dm[site].dropna().empty:
            continue
        st = daily_stats(site)
        s = dm[site][start.normalize():ends(site)].dropna()
        if s.empty:
            continue
        day = s.idxmax()
        if st.empty or (day.month, day.day) not in st.index:
            continue
        r = st.loc[(day.month, day.day)]
        rows.append(dict(site=site, dmax=s.max(), dday=day, p50=r.p50, p90=r.p90, p95=r.p95, dyears=int(r['count']),
                         day_max=r['max'], day_max_yr=r.max_yr))
    return pd.DataFrame(rows).set_index('site') if rows else pd.DataFrame()


def ranked(start, ends, before):
    """Event peaks (see event_peaks) with their standing in the annual-peak record before `before` (the event's
    water year): rank, years of record, previous record and its date, and the ratio to the median annual peak."""
    ev = event_peaks(start, ends)
    rows = []
    for site, r in ev.iterrows():
        pk = annual_peaks(site)
        pk = pk[pk.index < pd.Timestamp(before)]
        if len(pk) < 5 or r.peak <= 0:
            continue
        rows.append(dict(site=site, rank=int((pk > r.peak).sum() + 1), n=len(pk), first=int(pk.index.year.min()),
                         rec=pk.max(), rec_date=pk.idxmax(), median=pk.median(), ratio=r.peak / pk.median()))
    return ev.join(pd.DataFrame(rows).set_index('site'), how='inner')
