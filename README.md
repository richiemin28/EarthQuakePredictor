# Adaptive Earthquake Forecast System

An adaptive, continually-learning machine learning system for short-term earthquake
forecasting — currently covering **Myanmar** and **Japan**, two of the world's most
seismically active and consequential countries, with more regions plannable using the
same architecture.

The project began as the software artefact for an MSc dissertation on Myanmar
specifically (below), and has since been extended to a second country: Japan suffers
some of the most severe and frequent earthquakes on Earth (the 2011 Mw 9.0 Tohoku
earthquake among them) and has a far denser, better-instrumented public catalog than
Myanmar's — a genuinely useful, different test of whether the same continual-learning
approach holds up in a very different seismic and data regime. The goal is the same one
the dissertation started with: put a real, working, continuously-updating forecasting
tool in front of people living somewhere earthquakes are a real and recurring risk, not
just a one-off historical-data exercise.

> **Research artefact — not an early-warning system.** Predictions are probabilistic
> outputs of a statistical pattern-matching model, not seismological forecasts. They are
> not validated for operational use and must not be relied on for emergency response,
> public safety, or infrastructure decisions. See [Disclaimer](#disclaimer) below.

See [CHANGELOG.md](CHANGELOG.md) for release history.

---

## Why this exists

Myanmar sits at the junction of the Indian, Eurasian, and Sunda plates and is cut by the
Sagaing Fault, which accommodates roughly 20–23 mm/year of right-lateral strike-slip
motion. On 28 March 2025 an Mw 7.7 earthquake ruptured ~500 km of the fault near
Mandalay — one of the largest continental strike-slip events recorded globally in recent
decades. The seismic gap that produced it had been recognised in the literature for
years, yet no machine-learning-based prediction system existed for the region at the
time.

Despite that hazard, Myanmar is one of the least seismically monitored regions in Asia,
and virtually all published ML earthquake-prediction work (Mukherjee et al., 2025; Yavas
et al., 2024; Hu et al., 2025) shares the same limitation: models are trained once on a
fixed historical catalog and never updated, so their knowledge of evolving regional
seismicity grows steadily more stale. No published study had applied **continual
learning** to earthquake prediction in Myanmar, or any similarly data-sparse seismic
region, before this project.

**Hypothesis tested by this system:** a model that is continuously updated with incoming
real-time seismic data will outperform a static, frozen model at short-term earthquake
prediction, measured by precision, recall, F1, and AUC across multiple magnitude
thresholds and prediction windows. Myanmar was the original test case (data-sparse,
under-monitored); Japan is the second, deliberately different one (data-dense,
extremely well-monitored) — the same architecture, retrained independently on each
country's own catalog.

## What the system does

The system runs a controlled experiment between two models trained identically on
historical data, then diverging:

- **Static model** — frozen after initial training, never updated. Represents the
  conventional approach used throughout the literature.
- **Adaptive model** — after initial training, incrementally updates as new seismic
  events arrive, using **replay-based continual learning** (each update combines the new
  data with a randomly sampled 20% of prior training data) to mitigate catastrophic
  forgetting.

Both are XGBoost classifiers trained on 15 seismic indicator features computed with a
rolling 50-event window, following the feature taxonomy of Mukherjee et al. (2025).
**CTGAN** synthetic data augmentation (Joshi et al., 2025) balances the training set for
rare high-magnitude classes where needed. A dedicated spatial module turns raw
probability outputs into estimated epicenter coordinates, zone names, and uncertainty
radii using magnitude-weighted, recency-weighted spatial clustering across named
tectonic zones — ten for Myanmar (Sagaing Fault, Indo-Burman Range, Andaman Sea, etc.)
and ten for Japan (Nankai Trough, Japan Trench, Sagami Trough, etc.).

**The adaptive model keeps learning in production, not just in the evaluation.** A
daily scheduled job fetches confirmed new earthquakes for both countries, recomputes
features and labels, and genuinely retrains the adaptive model on them — see
[Keeping the model current](#keeping-the-model-current-in-production) below for exactly
how, since this is the kind of claim that's easy to leave true only on paper.

### Magnitude range differs by country, deliberately

Myanmar's catalog above M5.5 is thin — 182 events total over 1990–2019 — and the
dissertation's own evaluation already found M5.5 the hardest threshold to get reliable
signal at, so Myanmar's thresholds stop there (M4.5 / M5.0 / M5.5) rather than stretching
into thinner-still territory.

Japan's catalog carries real, trainable signal much further up the scale — 1,159 events
at M5.5+, 378 at M6.0+, 106 at M6.5+, 34 at M7.0+ over the same 30-year window, checked
against the USGS catalog directly before deciding this, not assumed. Below M5.5 many
Japan prediction windows were already saturating near 100% (which is a real result —
background seismicity there really is that dense at moderate magnitudes), which left the
higher-consequence range — where a major or great earthquake is more or less likely from
one window to the next — completely untracked. Japan's thresholds now run **M4.5 / M5.0
/ M5.5 / M6.0 / M6.5 / M7.0**, each confirmed to have enough real positive examples
(hundreds to low thousands, not just a handful) to be worth training on, not just
technically possible to fit a classifier to.

### Location precision: named zones only, tied to the forecast window, tightening over time

Every location prediction is a real named tectonic zone — Sagaing Fault Zone, Japan
Trench (Tohoku), and so on. There is no generic "Country Region (General)" catch-all in
the output; if recent activity isn't concentrated inside a known zone's boundary closely
enough to say something specific, the system reports the nearest well-established zone
by known seismic hazard significance rather than a vague region-sized answer.

Each zone's boundary is itself data-driven, not a rough administrative outline: it's the
magnitude-weighted middle 70% (15th–85th percentile) of real M≥3.0 events from the
1990–2025 catalog that fell within that zone's original, much larger boundary — the box
tightened down to where the real seismicity actually concentrates.

**Within a zone, the centroid comes from the actual hotspot, not an average across the
whole zone.** A zone can be 100-300km across, and recent events inside it don't
necessarily form one blob — averaging every event in the zone can land the "predicted"
point in a location with zero historical activity, geometrically the mean but
statistically meaningless (literally between two real clusters rather than at either
one). Recent events are density-clustered first (DBSCAN, haversine distance so it
respects the Earth's curvature — see `spatial_predictor._find_hotspot_mask`), and the
cluster with the most combined magnitude/recency weight — not just the most points —
becomes the basis for the centroid, radius, and depth. Every prediction reports which
one happened (`is_hotspot` in the JSON, "Based on: a real cluster of nearby events" vs.
"a wider average" in the map popup), so the confidence behind the number is never hidden.
Falls back to the whole-zone average only when there aren't enough points, or events are
too scattered, for any real cluster to emerge.

The uncertainty radius (and depth — see below) is computed fresh from recent clustering
(not a fixed per-zone number), and — since a "7 days from now" answer and a "30 days
from now" answer shouldn't necessarily point at the same circle — it's computed
separately per forecast window, with shorter windows drawing on a shorter, more
immediate lookback so a near-term forecast reflects near-term activity specifically.

**The precision floor scales with how much recent data actually backs the estimate,
down to 5km at the tightest — it is not one fixed number applied regardless of sample
size.** A tight radius computed from 2-3 events could just be coincidence, not genuine
confidence, so it isn't allowed to claim the same precision as the same tight number
backed by dozens of clustered events:

| Recent events behind the estimate | Radius floor | Depth floor |
|---|---|---|
| < 15 | 30 km | 10 km |
| 15–24 | 20 km | 8 km |
| 25–39 | 10 km | 5 km |
| ≥ 40 | 5 km | 3 km |

None of today's zones have enough near-term clustering (per the per-window lookback
above) to reach the tightest tier yet — that's expected, not a shortfall to hide. The
calculation is simply never artificially prevented from getting there as the adaptive
models accumulate more confirmed real earthquakes to cluster on, recomputed against a
recurring schedule (see
[Keeping the model current](#keeping-the-model-current-in-production)). It's a direct
extension of the same continual-learning story as the magnitude models: more confirmed
real data should mean the location answer keeps getting tighter, not that it was
hardcoded tight from day one.

**Depth is estimated the same way location is.** Alongside the horizontal centroid and
radius, each prediction now also reports an estimated depth and depth uncertainty —
computed identically (magnitude/recency-weighted mean and spread of recent nearby
events' recorded depths), floored by the same confidence tiers, shown in the map popup
for the top zone. Like the horizontal position, this is a descriptive statistic over
recent real events, not a physically modelled rupture depth.

## Result: does adaptive updating actually help?

Evaluated pseudo-prospectively across the 2020–2025 test period (1,279 held-out Myanmar
events, never seen during training):

| Magnitude threshold | What happened |
|---|---|
| **M5.5** (large, disaster-relevant events) | The static model scores **near-zero F1** and **below-random AUC** on 3 of 4 prediction windows — it is actively worse than guessing. The adaptive model recovers meaningful skill: **F1 0.15–0.43**, **AUC 0.59–0.66**. |
| **M5.0** | Adaptive AUC improves by **+0.08 to +0.19** across all four prediction windows. |
| **M4.5** | Adaptive model improves both F1 and AUC on all three windows evaluated. |
| **M4.0** | Both models score >0.98 F1 — this just reflects Myanmar's near-continuous background seismicity at that threshold and isn't a meaningful signal either way. |

Across the 12 meaningfully-evaluated label/window combinations (excluding the trivial
M4.0 case), the adaptive model beats the static baseline on **9/12 F1** and **10/12
AUC** comparisons. The improvement is not perfectly monotonic year-over-year (2023 shows
a temporary dip, consistent with known replay-buffer stability/plasticity trade-offs),
but the combined multi-year comparison is unambiguous, and the effect is largest exactly
where it matters most: detecting the region's most dangerous earthquakes. Full
methodology, per-year tables, and discussion are in Chapters 5–6 of the original
dissertation (Myanmar-only at time of writing).

Japan's own pseudo-prospective evaluation runs the identical methodology independently
on Japan's catalog — see `predictions/japan_latest_predictions.json` and the terminal
output of `python run_pipeline.py japan --mode evaluate` for current numbers; a
dedicated write-up comparing the two countries' results is a natural next step but isn't
folded into the original dissertation text, which predates Japan's addition.

## Architecture

```
config.py                Myanmar configuration — geographic bounds, API endpoints,
                          zones, magnitude thresholds, and file paths
config_japan.py           Japan's equivalent configuration, same structure, different
                          bounds/zones/thresholds/file paths (all japan_-prefixed so
                          nothing collides with Myanmar's data)
run_pipeline.py           Runs main.py or generate.py against a specific country's
                          config by swapping sys.modules["config"] before import -
                          every pipeline module does `from config import X`, so this
                          works without editing any of them. See its own docstring.
data_acquisition.py       USGS historical catalog fetch + real-time ATOM feed polling
feature_engineering.py    15 seismic indicator features + binary label construction
data_augmentation.py      CTGAN synthetic minority-class generation
models.py                 StaticModel and AdaptiveModel (XGBoost + replay buffer)
spatial_predictor.py      Magnitude/recency-weighted spatial clustering → zone,
                          centroid, depth, and uncertainty radius (confidence-scaled
                          by event count). Computed per forecast window (not one
                          fixed snapshot); named zones only, no generic catch-all
prediction_engine.py      Turns model output + spatial stats into structured
                          predictions, including per-magnitude effect/action text
live_updater.py           LiveUpdater class used by `main.py --mode live`
live_demo.py              Standalone real-time terminal dashboard (continuous polling
                          + adaptive updates + prediction refresh)
generate.py                Lightweight refresh: loads the saved model and regenerates
                          predictions without retraining (~60s) — used by CI
main.py                   Master pipeline entry point (train/update/evaluate/live/predict)
index.html                 Static front-end dashboard — mobile-first, light/dark theme,
                          country switcher, plain-language forecast summary
.github/workflows/
  update-predictions.yml   Every 6 hours: refreshes predictions for both countries from
                          the current model and deploys them, no retraining
  adaptive-update.yml      Once daily: the real continual-learning step — fetches
                          confirmed new earthquakes and actually retrains both
                          countries' adaptive models on them
```

Data flows: `data_acquisition.py` → `feature_engineering.py` → (`data_augmentation.py` →)
`models.py` → `spatial_predictor.py` / `prediction_engine.py` →
`predictions/{country}_latest_predictions.json` → `index.html`.

## Getting started

Requires Python 3.10+ (developed and tested on 3.13).

```bash
git clone https://github.com/richiemin28/EarthQuakePredictor.git
cd EarthQuakePredictor
pip install -r requirements.txt
```

`requirements.txt` covers the full pipeline (including CTGAN/`sdv`, used only for
training). `requirements_server.txt` is the trimmed dependency set used by CI (both the
lightweight refresh and the real adaptive-update job — neither needs CTGAN).

### Run the full pipeline

```bash
# Myanmar (default config):
python main.py --mode full

# Japan (or any future country with its own config_<name>.py):
python run_pipeline.py japan --mode full
```

This takes a while on first run — the historical fetch and rolling-window feature
computation are the slow steps. Results are cached to `data/` and `models/` (with
`japan_`-prefixed filenames for Japan) so subsequent runs are fast.

### Other modes

Works identically for either country — just choose whether to call `main.py` directly
(Myanmar) or route it through `run_pipeline.py <country>` (any other country):

```bash
python main.py --mode train              # (Re)train static + adaptive models from scratch
python main.py --mode train --refresh    # Force a fresh USGS download instead of using the cache
python main.py --mode evaluate           # Pseudo-prospective evaluation, static vs adaptive
python main.py --mode predict            # Generate structured predictions from the saved model
python main.py --mode update             # Fetch new confirmed events, actually retrain the model
python main.py --mode live               # Start the continuous live-updating loop

python run_pipeline.py japan --mode train
python run_pipeline.py japan --mode update
python run_pipeline.py japan generate    # equivalent to generate.py, but for Japan
```

### Live terminal dashboard

```bash
python live_demo.py                 # Myanmar, poll USGS every 5 minutes (default)
python live_demo.py --interval 60   # Poll every 60 seconds
```

Polls the USGS ATOM feed, ingests any new events in the bounding box, triggers an
adaptive model update, and redraws a full prediction dashboard (recent activity, zone
rankings, forward predictions with coordinates/radius/probability) in the terminal. Press
`Ctrl+C` to stop — the model checkpoints automatically on exit.

### Web dashboard

`index.html` is a self-contained static page — open it directly in a browser, or serve it
with any static file server. It reads `predictions/{country}_latest_predictions.json`
for the ML forecasts and polls the USGS ATOM feed directly in the browser for live
activity. A country switcher in the header lets a visitor pick Myanmar or Japan without
reloading the page; a light/dark theme toggle persists the visitor's choice.

Mobile-first: base styles are a normal scrolling single-column page with 44px+ touch
targets, and the fixed two-pane app-shell layout only kicks in above 900px wide. A
plain-language "hero" summary (the single highest 30-day probability, described in
words — "78% chance of a strong earthquake near the Sagaing Fault Zone" — not just a
raw percentage) leads every visit; the full multi-window forecast table is available
behind a "see full forecast" disclosure for anyone who wants the detail.

## Keeping the model current in production

Two separate scheduled jobs, deliberately different cadences:

- **`update-predictions.yml`, every 6 hours** — cheap. Loads whatever model already
  exists and regenerates predictions using the latest 90 days of USGS data for spatial
  context. Doesn't touch the model's learned weights.
- **`adaptive-update.yml`, once daily at 03:00 UTC** — the real thing. Fetches every
  confirmed new earthquake since the last run, recomputes the full feature/label set,
  and actually calls the adaptive model's `.update()` method — genuine incremental
  retraining on real outcomes, with the replay buffer doing its job of not overwriting
  what the model already knew. Runs daily rather than every 6 hours deliberately:
  continual learning is more stable with a meaningful batch of new events per update
  than with many very small ones. Both countries' catalogs, models, and predictions are
  committed back to the repo and redeployed after each run.

Both workflows share a `concurrency` group so they can never run at the same time and
race to push to `main` — GitHub queues one behind the other instead. If you fork this
repo, `workflow_dispatch` lets you trigger either one manually from the Actions tab to
see it work without waiting for the schedule.

## Data sources

- **[USGS Earthquake Catalog API](https://earthquake.usgs.gov/fdsnws/event/1/)** —
  FDSN-compliant historical catalog. Myanmar: M2.0+, 10–30°N / 90–105°E. Japan: M4.5+
  (Japan's catalog is dense enough that M2.0 would make feature computation
  intractable), 24–46°N / 122–146°E.
- **[USGS ATOM feed](https://earthquake.usgs.gov/earthquakes/feed/)** — real-time event
  stream, polled for live updates.

Both are free, public, and require no API key.

## Limitations

Carried over from the dissertation's own assessment (Chapter 6.2), plus what's changed
since Japan was added — worth reading before trusting any output:

- **Catalog sparsity (Myanmar specifically).** Myanmar's public USGS catalog
  under-represents actual seismicity compared to well-monitored regions; a dedicated
  seismic array study (Yang et al., 2024) found roughly double the events of routine
  processing. Japan's catalog doesn't share this limitation — it's one of the
  best-instrumented seismic regions on Earth.
- **Reduced feature set.** 15 of the 61 features used by the reference study (Mukherjee
  et al., 2025) are implemented here, chosen by their reported SHAP importance.
- **Simple continual-learning strategy.** Fixed-ratio replay buffering is the simplest
  viable catastrophic-forgetting mitigation; it produces a non-monotonic
  year-by-year trajectory (a temporary dip in 2023 for Myanmar) rather than uniformly
  increasing performance.
- **Location estimates are descriptive, not physical.** Coordinates and radii come from
  clustering of recent seismicity, not from fault-geometry or stress modelling — they show
  *where recent activity has concentrated*, not a physically derived rupture probability.
- **Japan's higher-magnitude thresholds (M6.5, M7.0) rest on real but comparatively
  fewer positive examples** than the lower thresholds, even though each was checked to
  clear the minimum needed for reliable cross-validation before being included (see
  [Magnitude range differs by country](#magnitude-range-differs-by-country-deliberately)).
  Treat the higher end of Japan's range as directionally meaningful, not as precise as
  the well-populated M4.5–5.5 range.
- **No operational deployment.** This has been validated in pseudo-prospective evaluation
  and confirmed to run correctly in live polling mode, but has not been reviewed by
  seismologists or integrated with any civil protection framework, for either country.

## Disclaimer

This is an experimental research artefact developed for academic purposes only. Its
probabilistic predictions are based on statistical pattern analysis of historical and
real-time seismic catalog data and are **not validated for operational use**. They do
not constitute official earthquake warnings, hazard assessments, or civil protection
advisories of any kind, for Myanmar, Japan, or any other region. No reliance should be
placed on the system's outputs for emergency response, public safety, or infrastructure
planning without independent validation by qualified seismologists and integration with
established civil protection frameworks. Deterministic earthquake prediction (precise
time, location, and magnitude) remains beyond the reach of current science — this system
produces probabilistic forecasts within defined spatial, temporal, and magnitude windows,
nothing more.

## Citation

If you build on this work, please cite the original dissertation, which this project
extends:

```
Min Pyae (2026). AI Based Real Time Earthquake Prediction for Myanmar and Surrounding
Seismic Regions Using Continual Learning. MSc Dissertation, University of Chester.
```

## Key references

The methodology draws directly on the following published work — see the dissertation's
reference list for the complete set:

- Mukherjee, B., Shaw, R. L., Sharma, M. L., & Sain, K. (2025). Earthquake prediction
  using machine learning perspectives in Himalayan seismic belt and its surroundings.
  *Journal of Asian Earth Sciences, 293*, 106764.
- Dascher-Cousineau, K., Shchur, O., Brodsky, E. E., & Günnemann, S. (2023). Using deep
  learning for flexible and scalable earthquake forecasting. *Geophysical Research
  Letters, 50*, e2023GL103909.
- Van de Ven, G. M., Tuytelaars, T., & Tolias, G. (2024). Continual learning and
  catastrophic forgetting. arXiv:2403.05175.
- Joshi, A., Raman, B., & Mohan, C. K. (2025). Real time earthquake magnitude prediction
  using designed machine learning ensemble trained on real and CTGAN generated synthetic
  data. *Geodesy and Geodynamics, 16*, 350–368.
- Devi, S., Pasari, S., & Mehta, A. (2025). Seismic cycle progression in major cities of
  Myanmar using earthquake nowcasting. *Journal of Seismology*.
- Yang, S. et al. (2024). New insights into active faults revealed by a deep learning
  based earthquake catalog in central Myanmar. *Geophysical Research Letters, 51*,
  e2023GL105159.

## Acknowledgements

Thanks to Dr Joe Collenette for supervising the original Myanmar dissertation this
project is built on, and to the **United States Geological Survey** for maintaining
free, open access to the earthquake catalog API and real-time ATOM feed that the entire
project — Myanmar, Japan, and any country added after — runs on. All research design,
experimental decisions, and analysis are the author's own.

## License

No license file is currently included in this repository — all rights are reserved by
default under copyright law. If you'd like to reuse, modify, or redistribute this code,
please open an issue or contact the author first.
