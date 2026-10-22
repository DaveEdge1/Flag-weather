"""Shared settings: paths, sites, map domain, thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'            # download cache (git-ignored except data/derived)
DERIVED = DATA / 'derived'      # small processed tables that are committed
for _d in (DATA, DERIVED):
    _d.mkdir(exist_ok=True)

FLAGSTAFF = dict(lat=35.2, lon=-111.65)

# NWS Flagstaff Pulliam Airport, threaded record (RCC-ACIS)
ACIS_SID = 'FLGthr'

# NRCS SNOTEL sites (station triplets)
SNOTEL = {
    '1121:AZ:SNTL': 'Fort Valley',
    '927:AZ:SNTL': 'Snowslide Canyon',
    '488:AZ:SNTL': 'Fry',
}

# ERA5 download domain [N, W, S, E]: North Pacific to the Southwest US
ERA5_AREA = [60, -180, 10, -95]

AR_IVT = 250  # kg m-1 s-1, conventional atmospheric-river threshold
