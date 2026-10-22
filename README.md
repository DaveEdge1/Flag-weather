# Flag-weather

Storm events and monthly climate synopses for Flagstaff, Arizona: https://daveedge1.github.io/Flag-weather/

Each **event page** pairs rainfall across the City of Flagstaff gauge network (and nearby SNOTEL sites) with ERA5 maps of moisture transport (IVT), the 250-hPa jet and 500-hPa heights. Each **monthly synopsis** compares temperature and precipitation with normal and links the month's anomalies to its events and to climate indices.

## Layout

```
flagweather/          Python package
  config.py           paths, sites, ERA5 domain, thresholds
  sources/            data access: gauges (City ALERT), snotel (NRCS AWDB), acis (airport), era5 (CDS), nhc (best track)
  figures/            event maps and time series
site/                 Quarto website (rendered by GitHub Actions to GitHub Pages)
data/                 download cache (git-ignored); data/derived/ holds small committed tables
```

## Setup

```
pip install -r requirements.txt
```

ERA5 downloads need a Copernicus CDS account: put your API token in `~/.cdsapirc` and accept the ERA5 licences on the CDS website. The GitHub Actions build does not download ERA5; event figures are rendered locally and frozen.

To preview the site: `quarto preview site`

## Data and attribution

See the [Methods](https://daveedge1.github.io/Flag-weather/methods.html) page. Rain-gauge data courtesy of the City of Flagstaff (provisional). Contains modified Copernicus Climate Change Service information. SNOTEL: USDA NRCS. Airport: NOAA NWS via RCC-ACIS. Tropical cyclones: NOAA NHC.

Code is released under the MIT licence.
