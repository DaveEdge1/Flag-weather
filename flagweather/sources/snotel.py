"""NRCS SNOTEL daily data via the AWDB REST API."""
import json
import time
import urllib.request

import pandas as pd

from ..config import DERIVED

API = 'https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1'
# WTEQ snow water equivalent (in), SNWD snow depth (in), PREC water-year accumulated precip (in),
# TMAX/TMIN/TAVG air temperature (degF)
ELEMENTS = ['WTEQ', 'SNWD', 'PREC', 'TMAX', 'TMIN', 'TAVG']


def _retry(fn, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def _get(path):
    req = urllib.request.Request(API + path, headers={'User-Agent': 'Flag-weather'})
    return _retry(lambda: json.loads(urllib.request.urlopen(req, timeout=45).read().decode()))


def stations(triplets):
    """Station metadata; cached in data/derived (committed) because it rarely changes."""
    f = DERIVED / 'snotel_stations.csv'
    if f.exists():
        cached = pd.read_csv(f, index_col='triplet', dtype={'begin': str})
        if set(triplets) <= set(cached.index):
            return cached.loc[list(triplets)]
    d = _get('/stations?stationTriplets=' + ','.join(triplets) + '&returnStationElements=false')
    df = pd.DataFrame([dict(triplet=s['stationTriplet'], name=s['name'], lat=s['latitude'], lon=s['longitude'],
                            elev_ft=s['elevation'], begin=s.get('beginDate', '')[:10]) for s in d]).set_index('triplet')
    df.to_csv(f)
    return df


def daily(triplets, start, end, elements=ELEMENTS):
    """Long table: date, triplet, element, value, median (1991-2020 median for that date, where defined)."""
    d = _get(f'/data?stationTriplets={",".join(triplets)}&elements={",".join(elements)}'
             f'&duration=DAILY&beginDate={start}&endDate={end}&centralTendencyType=MEDIAN')
    rows = [dict(date=pd.Timestamp(v['date']), triplet=s['stationTriplet'],
                 element=e['stationElement']['elementCode'], value=v.get('value'), median=v.get('median'))
            for s in d for e in s['data'] for v in e['values']]
    return pd.DataFrame(rows)
