# Changelog

## v1.5.1 — Fix the full forecast table silently overlapping the disclaimer

### Fixed

- **The full forecast table (every magnitude x every time window) was
  visually overlapping the disclaimer below it on every single mobile
  and desktop visit, for both countries.** Same root cause as the
  v1.5.0 window-toggle fix: `.details-card` was a direct auto-sized
  flex child of `.pane` (a column-direction flex container) with
  `overflow:hidden` set, which collapsed its own box to ~2px tall while
  its content (the whole matrix table, 471px+ of it) kept rendering at
  its real position - so everything below it in the document, starting
  with the disclaimer, laid out as if the table took up almost no space
  and ended up drawn on top of it instead of below it. It also meant
  the panel's true scrollable height came up ~470px short, so scrolling
  to "the bottom" on mobile never actually reached the real bottom.
  Found by deliberately scrolling a mobile viewport to the pane's
  reported `scrollHeight` and noticing the matrix simply wasn't there -
  it had been rendering "correctly" in every prior screenshot only
  because those screenshots happened not to scroll past the collapse
  point. Fixed the same way as the toggle: dropped the `overflow:hidden`
  (none of this card's children have their own background near the
  rounded corners, so nothing else needed to change).
- Audited every other `overflow:hidden` rule in the stylesheet against
  the same failure pattern (direct, auto-sized flex child of a
  column-direction flex container - `#main` and `.pred-zone-card` looked
  similar at a glance but aren't actually at risk: `#main` has explicit
  `flex:1`, not auto-sizing, and `.pred-zone-card` isn't a *direct* flex
  child of a flex container, it's nested inside a plain div). No other
  instances found.

---

## v1.5.0 — Switchable forecast window

### Added

- **The hero card and the bars below it are no longer locked to 30 days.**
  A 7d/15d/30d switcher above the hero card updates both together -
  headline percentage, "next N days" wording, and the bars' own title and
  breakdown all follow whichever window is selected. Persists across
  reloads and country switches the same way the theme and country choice
  already did.
- Added a `MAGNITUDE_INFO` entry for M4.0 (Myanmar-only), fixing a gap
  where its hero text fell back to the raw `M4.0+` label instead of a
  proper word - only became visible once the window switcher could
  surface M4.0 as the top prediction for short windows.

### Fixed

- A layout bug where the new window-toggle control collapsed to ~2px
  tall and was effectively invisible: `overflow:hidden` combined with
  `display:inline-flex` collapsed its auto-height in this specific
  nested-flex context (a flex item that's itself a flex container, inside
  a column-direction flex parent) - reproducible in Chrome 151, didn't
  reduce to any single property in isolation. Sidestepped by rounding the
  end buttons directly instead of clipping the container's overflow.

---

## v1.4.0 — Full M4-M10 range, no more collapsing predictions down to one

### Added

- **Myanmar's magnitude range extended to M6.0**, and M4.0 (already trained,
  previously hidden from display) is shown again. M6.0 was added after
  checking real row-level label counts - 59 independent M≥6.0 events over
  1990-2025, clearing the same bar Japan's own thresholds were held to.
- **The dashboard now shows the full M4-M10 range for both countries** -
  every threshold with enough real historical events gets a genuine
  forecast; every threshold without one is shown too, explicitly marked
  "insufficient historical data" instead of hidden or faked. Checked real
  event counts before drawing the line: Myanmar stops at M6.0 (M6.5+ has
  only 18 real events, M7.0+ just 3), Japan stays at M7.0 (M7.5+ has only
  10, M8.0+ is essentially the 2011 Tohoku earthquake alone). Going all
  the way to M10 with real classifiers isn't possible for either catalog -
  training on 1-2 historical events isn't learning a pattern, it's fitting
  one earthquake, regardless of how healthy the rolling-window label counts
  might look (a single massive quake makes every preceding day's row count
  as "positive", which can disguise n=1 as n=100+ if you only check label
  counts and not real independent event counts).

### Changed

- **Predictions are no longer collapsed down to a single number per zone.**
  Every zone card now shows its full probability ladder across every
  magnitude threshold (e.g. "M4 100% · M4.5 100% · M5 86% · M5.5 48% ·
  M6 1%"), not just whichever single prediction happened to be highest.
  The map popup got the same treatment. The full forecast table (every
  magnitude x every time window) is now open by default instead of
  tucked behind a click.

---

## v1.3.0 — Cluster-based location, not zone-wide averaging

### Changed

- **The predicted centroid now comes from the zone's actual hotspot, not an
  average across the whole zone.** A named zone can be 100-300km across,
  and recent events inside it don't necessarily form one blob - averaging
  every event in the zone could land the "predicted" point in a location
  with zero historical activity: geometrically the mean, statistically
  meaningless, literally between two real clusters rather than at either
  one. Recent events are now density-clustered first (`sklearn.cluster.
  DBSCAN`, haversine distance so it respects the Earth's curvature rather
  than treating degrees of latitude/longitude as equivalent), and the
  cluster with the most combined magnitude/recency weight - not just the
  most points - becomes the basis for the centroid, radius, and depth.
  Falls back to the old whole-zone average only when there aren't enough
  points, or events are too scattered, for a real cluster to emerge.
- Every location prediction now reports which basis it used
  (`primary_is_hotspot` in the JSON; "Based on: a real cluster of nearby
  events" vs. "a wider average" in the map popup) - the confidence behind
  the number is never hidden.
- Zone ranking gives a modest bonus to hotspot-backed predictions over
  whole-zone-average ones when zones are otherwise close in score, since a
  real cluster is a more trustworthy basis for "where" than an average is.
- Effect on live numbers: Myanmar's top zone (Sagaing Fault Zone) dropped
  from ~83km to 30km (the confidence-tier floor for its now-3-event
  cluster, down from averaging all 6 zone-wide events); Japan's ranking
  shifted to Median Tectonic Line (SW Japan) at 30km, whose 4 recent events
  were already a single tight cluster, ahead of Japan Trench (Tohoku)
  whose 6 events split into a smaller 3-event cluster once genuinely
  distinct activity was separated out.

---

## v1.2.0 — Depth prediction, confidence-scaled precision, mobile landscape fix

### Added

- **Depth prediction.** Every location prediction now includes an estimated
  depth and depth uncertainty (`primary_depth_km` / `primary_depth_range_km`
  in the JSON, shown in the map popup), computed the same way the
  horizontal location already was: magnitude/recency-weighted mean and
  spread of recent nearby events' recorded depths. Same descriptive-
  statistic honesty as the horizontal radius - not a physically modelled
  rupture depth.

### Changed

- **The uncertainty radius floor is now confidence-scaled, not one fixed
  number.** v1.1.0 lowered the floor from 50km to a flat 30km. It's now
  tiered by how many recent events actually back the estimate - as low as
  **5km** when a zone has 40+ well-clustered recent events, loosening to
  30km when there are fewer than 15. A tight radius from 2-3 events could
  just be coincidence, not genuine confidence; it shouldn't claim the same
  precision as the same tight number backed by dozens of events. Depth
  uncertainty uses the same tiering (3km down to 10km). See the README's
  location precision section for the full table.

### Fixed

- **Mobile landscape orientation was showing almost nothing.** The bottom
  sheet's "peek" state was a fixed percentage of available height with no
  minimum - on a landscape phone (~390px tall), that collapsed to ~114px,
  not even enough to show the headline percentage, just the tab bar. Sheet
  states now have pixel floors alongside their percentages, and the header
  compacts itself on short viewports (`max-height: 500px`) to free up more
  room for the sheet in the first place.
- A stale map-popup label read "Recent events (90d)" - left over from
  before spatial stats moved to a per-forecast-window lookback (14-45 days,
  not a fixed 90). Now just "Recent nearby events", accurate regardless of
  which window is driving the display.

---

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
