# Flag-weather

Storm events and monthly climate synopses for Flagstaff, Arizona: https://daveedge1.github.io/Flag-weather/

Each **event page** pairs rainfall across the City of Flagstaff gauge network (and nearby SNOTEL sites) with ERA5 maps of moisture transport (IVT), the 250-hPa jet and 500-hPa heights. Each **monthly synopsis** compares temperature and precipitation with normal and links the month's anomalies to its events and to climate indices.

## Layout

```
flagweather/          Python package
  config.py           paths, sites, ERA5 domain, thresholds
  sources/            data access: gauges (City ALERT), snotel (NRCS AWDB), acis (airport), era5 (CDS), nhc (best track)
  event.py            event data, objective tags and summary numbers
  page.py             event-page text and tables
  figures/            event maps, gauge maps and time series
scripts/              make_event_page.py
site/                 Quarto website (rendered by GitHub Actions to GitHub Pages)
data/                 download cache (git-ignored); data/derived/ holds small committed tables
```

## Setup

```
pip install -r requirements.txt
```

ERA5 downloads need a Copernicus CDS account: put your API token in `~/.cdsapirc` and accept the ERA5 licences on the CDS website. In GitHub Actions the token comes from the repository secret `CDS`.

## Adding an event

```
python -c "from flagweather.sources import era5; era5.fetch('2026-10-09', '2026-10-14')"   # if ERA5 covers it
python scripts/make_event_page.py 2026-10-09T12 2026-10-14T06 "October 9–13, 2026" [MAPTIMES...]
quarto render site
```

Start and end are UTC. Pages whose window ERA5 covers are marked Final and frozen (`site/_freeze/`, committed); others are Provisional and re-execute on each build, including the daily scheduled build.

To preview the site: `quarto preview site`

## Data and attribution

See the [Methods](https://daveedge1.github.io/Flag-weather/methods.html) page. Rain-gauge data courtesy of the City of Flagstaff and Coconino County, used with the City's permission for non-commercial purposes (provisional). Contains modified Copernicus Climate Change Service information. SNOTEL: USDA NRCS. Airport: NOAA NWS via RCC-ACIS. Tropical cyclones: NOAA NHC.

Code is released under the MIT licence.
