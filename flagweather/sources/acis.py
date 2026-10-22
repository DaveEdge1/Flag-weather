"""NWS station data via RCC-ACIS (Flagstaff Pulliam Airport threaded record)."""
import json
import urllib.request

import pandas as pd

from ..config import ACIS_SID

API = 'https://data.rcc-acis.org'


def _post(path, q):
    req = urllib.request.Request(API + path, data=json.dumps(q).encode(),
                                 headers={'Content-Type': 'application/json', 'User-Agent': 'Flag-weather'})
    return json.loads(urllib.request.urlopen(req, timeout=120).read().decode())


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
