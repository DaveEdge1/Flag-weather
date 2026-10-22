"""Natural-feature basemap for the Flagstaff gauge maps: USGS 3DEP shaded relief, elevation
contours, NHD streams, peaks located from the elevation data, and interstate, US and state
highways (Census TIGERweb).

The DEM, streams and roads are fetched once and stored in data/derived/ (committed), so site builds
do not depend on the USGS services.
"""
import json
import re
import urllib.parse
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

from .config import DERIVED

BBOX = (-111.90, 35.04, -111.45, 35.40)   # W, S, E, N
SIZE = (900, 720)
DEM_FILE = DERIVED / 'basemap_dem.npz'
STREAM_FILE = DERIVED / 'basemap_streams.geojson'
ROAD_FILE = DERIVED / 'basemap_roads.geojson'
DEM_URL = 'https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage'
NHD_URL = 'https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/4/query'
TIGER_URL = 'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Transportation/MapServer/{}/query'
# (label, approx. lat, lon, search radius deg, label offset pts, alignment); marker at the DEM maximum near the point
PEAKS = [('Humphreys Peak', 35.346, -111.678, 0.03, (-6, 12), 'right'),
         ('Mount Elden', 35.242, -111.597, 0.03, (5, -4), 'left')]


def fetch():
    import io

    from PIL import Image
    q = dict(bbox=','.join(map(str, BBOX)), bboxSR=4326, imageSR=4326, size=f'{SIZE[0]},{SIZE[1]}', format='tiff',
             pixelType='F32', noDataInterpretation='esriNoMatchAny', interpolation='RSP_BilinearInterpolation',
             f='image')
    raw = urllib.request.urlopen(DEM_URL + '?' + urllib.parse.urlencode(q), timeout=180).read()
    dem = np.array(Image.open(io.BytesIO(raw)), dtype=np.float32)
    np.savez_compressed(DEM_FILE, dem=np.round(dem).astype(np.int16), bbox=np.array(BBOX))
    q = dict(geometry=','.join(map(str, BBOX)), geometryType='esriGeometryEnvelope', inSR=4326,
             spatialRel='esriSpatialRelIntersects', outFields='GNIS_NAME,FTYPE,StreamOrde',
             returnGeometry='true', outSR=4326, f='geojson')
    gj = urllib.request.urlopen(NHD_URL + '?' + urllib.parse.urlencode(q), timeout=180).read().decode()
    STREAM_FILE.write_text(gj, encoding='utf-8')
    feats = []
    for layer in (2, 6):   # primary roads; secondary roads (72k scale)
        q = dict(geometry=','.join(map(str, BBOX)), geometryType='esriGeometryEnvelope', inSR=4326,
                 spatialRel='esriSpatialRelIntersects', outFields='NAME,MTFCC,RTTYP', returnGeometry='true',
                 outSR=4326, f='geojson')
        d = json.loads(urllib.request.urlopen(TIGER_URL.format(layer) + '?' + urllib.parse.urlencode(q),
                                              timeout=180).read().decode())
        feats += [f for f in d.get('features', []) if _route(f['properties'])]
    ROAD_FILE.write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}), encoding='utf-8')


def _route(p):
    """Short label for interstate (I), US (U) and state (S) routes; None for business/historic routes."""
    name, typ = (p.get('NAME') or ''), p.get('RTTYP')
    if typ not in ('I', 'U', 'S') or 'Bus' in name or name.startswith('Hst'):
        return None
    num = re.search(r'(\d+\w*)', name)
    return f"{dict(I='I-', U='US ', S='SR ')[typ]}{num.group(1)}" if num else None


def load():
    if not (DEM_FILE.exists() and STREAM_FILE.exists() and ROAD_FILE.exists()):
        fetch()
    d = np.load(DEM_FILE)
    return (d['dem'].astype(float), tuple(d['bbox']), json.loads(STREAM_FILE.read_text(encoding='utf-8')),
            json.loads(ROAD_FILE.read_text(encoding='utf-8')))


def draw(ax, extent=None):
    """Draw relief, 200-m contours, streams and peak labels; returns the elevation extent drawn."""
    dem, (w, s, e, n), streams, roads = load()
    lons = np.linspace(w, e, dem.shape[1])
    lats = np.linspace(n, s, dem.shape[0])
    ls = LightSource(azdeg=315, altdeg=40)
    shade = ls.hillshade(dem, vert_exag=2, dx=30, dy=30)
    ax.imshow(shade, extent=(w, e, s, n), cmap='gray', vmin=-0.2, vmax=1.3, alpha=0.55, zorder=0,
              interpolation='bilinear', aspect='auto')
    cs = ax.contour(lons, lats, dem, np.arange(2000, 3900, 200), colors='#6f6d68', linewidths=0.35, alpha=0.6, zorder=0.5)
    ax.clabel(cs, cs.levels[::2], fontsize=6, fmt=lambda v: f'{v:.0f} m', inline_spacing=2)
    named = {}
    for f in streams.get('features', []):
        g = f.get('geometry') or {}
        parts = g.get('coordinates', [])
        if g.get('type') == 'LineString':
            parts = [parts]
        for p in parts:
            x, y = zip(*p)
            ax.plot(x, y, color='#5598e7', lw=0.8, alpha=0.8, zorder=1)
        name = (f['properties'].get('GNIS_NAME') or '').strip()
        if name and parts:
            named.setdefault(name, []).extend(parts[0])
    for name, pts in named.items():
        if len(pts) >= 20:   # label longer named streams once, at their midpoint
            x, y = pts[len(pts) // 2]
            if w < x < e and s < y < n:
                ax.annotate(name, (x, y), fontsize=6.5, color='#1c5cab', style='italic', zorder=2,
                            xytext=(3, -8), textcoords='offset points')
    labelled = set()
    for f in sorted(roads['features'], key=lambda f: f['properties'].get('RTTYP') != 'I'):
        lab = _route(f['properties'])
        g = f['geometry']
        parts = g['coordinates'] if g['type'] == 'MultiLineString' else [g['coordinates']]
        inter = f['properties']['RTTYP'] == 'I'
        for p in parts:
            x, y = zip(*p)
            ax.plot(x, y, color='#3a3936' if inter else '#6f6d68', lw=1.8 if inter else 1.1, alpha=0.85,
                    zorder=1.5, solid_capstyle='round')
        if lab not in labelled and parts:
            p = max(parts, key=len)
            x, y = p[len(p) // 2]
            if w < x < e and s < y < n:
                ax.annotate(lab, (x, y), fontsize=7, color='#0b0b0b', ha='center', va='center', zorder=2.5,
                            bbox=dict(boxstyle='round,pad=0.2', fc='#fcfcfb', ec='#3a3936', lw=0.6))
                labelled.add(lab)
    for label, la, lo, r, off, ha in PEAKS:
        m = (np.abs(lats[:, None] - la) <= r) & (np.abs(lons[None, :] - lo) <= r)
        i = np.unravel_index(np.argmax(np.where(m, dem, -np.inf)), dem.shape)
        ax.plot(lons[i[1]], lats[i[0]], marker='^', ms=5, color='#52514e', zorder=2)
        ax.annotate(label, (lons[i[1]], lats[i[0]]), xytext=off, ha=ha, textcoords='offset points', fontsize=7.5,
                    color='#3a3936', style='italic', zorder=2)
    ax.set_xlim(*(extent[:2] if extent else (w, e)))
    ax.set_ylim(*(extent[2:] if extent else (s, n)))
