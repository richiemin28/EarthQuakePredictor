# Changelog

## v1.1.0 — Precise, evolving location predictions

Follow-up to v1.0.0's multi-country stable release, focused entirely on the
location side of the forecast: how precise it is, how honest it is about
that precision, and how it's set up to keep improving.

### Removed

- **The generic "Myanmar Region (General)" / "Japan Region (General)"
  catch-all can no longer appear as a location prediction.** It's excluded
  from zone ranking entirely now, not just penalised — every prediction
  points at a real, named tectonic zone (Sagaing Fault Zone, Japan Trench
  (Tohoku), Nankai Trough, etc.) or, if recent activity genuinely doesn't
  support a specific answer, the nearest well-established zone by known
  seismic hazard significance. Never a vague, country-sized circle.

### Changed

- **Zone boundaries are tighter and data-driven.** Every one of the 20
  named zones (10 per country) was redrawn to the magnitude-weighted middle
  70% (15th-85th percentile) of real M>=3.0 events from the 1990-2025
  catalog that fell inside the zone's original, much larger boundary -
  tightened to where the seismicity actually concentrates, not an
  administrative-region-sized box.
- **The uncertainty radius is now computed per forecast window, not once
  per generation run.** A 7-day forecast and a 30-day forecast used to
  show the exact same circle because spatial stats were computed once
  with a fixed 90-day lookback and reused everywhere. Now each window
  gets its own lookback (roughly 1.5x the window, floored at 14 days,
  capped at 45), so a near-term forecast reflects near-term clustering
  specifically.
- **Precision floor lowered from 50km to 30km.** 30km is the target this
  system is working toward as its adaptive models accumulate more
  confirmed real earthquakes to cluster on, not a number already reached
  everywhere - the calculation is simply no longer artificially prevented
  from getting there once a zone's real data supports it.
- Live radius for the current top zone: **~70-83km** (down from ~81-108km
  after the first tightening pass, ~400-500km before any of this work).

### Fixed

- Japan's zone-ranking priority table only had Myanmar's zone names, so
  Japan's real fault zones never got a ranking bonus over the catch-all -
  each country now has its own `ZONE_PRIORITY` table in its own config.
- The "no ranked zones" fallback path (used when a country has no recent
  named-zone activity at all) was hardcoded to Myanmar's zone names
  regardless of which country's pipeline was running - it now derives the
  fallback from that country's own `ZONE_PRIORITY` table.
- The sparse-data fallback inside the spatial clustering calculation used
  to abandon the requested lookback entirely and grab the last 200 events
  by raw count whenever a region was quiet - untethered from any real time
  window, so a quiet stretch could silently pull in over a year of history
  for what was meant to be a 7-day-relevant answer. Replaced with
  progressive time-window expansion (2x, 4x, 90 days, 180 days), so it's
  always still a bounded, real lookback.
- `identify_zone()`'s fallback zone name was hardcoded to "Myanmar Region
  (General)" regardless of country (moot now that the catch-all is
  excluded from predictions entirely, but was a real display bug on the
  way here - see above).

---

## v1.0.0 — Multi-country adaptive earthquake forecasting

First stable release. Started as a single-country (Myanmar) ML earthquake
forecast dashboard; this release added Japan as a second supported
country, closed the loop on continual learning in production, and
rebuilt the frontend for everyday, non-technical users on any device.

- Japan added as a second country, its own magnitude-threshold ladder
  (M4.5-M7.0) reflecting its higher-magnitude seismicity.
- Continual learning runs in production: a daily scheduled job updates
  both countries' adaptive models against new USGS data with a
  replay-buffer strategy.
- Full UI redesign: monochrome base theme, precise 5-band probability
  scale, plain-language confidence note, light/dark mode, mobile-first
  responsive layout.
- Two scheduled GitHub Actions workflows keep the site live: predictions
  refresh every 6 hours, adaptive model updates once a day.
