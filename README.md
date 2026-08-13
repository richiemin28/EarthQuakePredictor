# Myanmar Earthquake Prediction System

An adaptive, continually-learning machine learning system for short-term earthquake
forecasting in Myanmar and its surrounding tectonic region (Sagaing Fault, Indo-Burman
Range, Andaman Sea subduction zone, Yunnan border, northern Thailand).

This repository is the software artefact for the MSc dissertation *"AI Based Real Time
Earthquake Prediction for Myanmar and Surrounding Seismic Regions Using Continual
Learning"* (Min Pyae, MSc Advanced Computer Science, University of Chester, CO7047
Research Project, supervised by Dr Joe Collenette, May 2026). Everything below — the
motivation, the architecture, and the reported results — reflects what is documented in
that report, so the two stay in sync.

> **Research artefact — not an early-warning system.** Predictions are probabilistic
> outputs of a statistical pattern-matching model, not seismological forecasts. They are
> not validated for operational use and must not be relied on for emergency response,
> public safety, or infrastructure decisions. See [Disclaimer](#disclaimer) below.

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
prediction for Myanmar, measured by precision, recall, F1, and AUC across multiple
magnitude thresholds and prediction windows.

## What the system does

The system runs a controlled experiment between two models trained identically on 1990–2019
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
rare high-magnitude classes. A dedicated spatial module turns raw probability outputs
into estimated epicenter coordinates, zone names, and uncertainty radii using
magnitude-weighted, recency-weighted spatial clustering across ten named tectonic zones.

## Result: does adaptive updating actually help?

Evaluated pseudo-prospectively across the 2020–2025 test period (1,279 held-out events,
never seen during training):

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
methodology, per-year tables, and discussion are in Chapters 5–6 of the dissertation.

## Architecture

```
config.py               Central configuration — geographic bounds, API endpoints,
                         zones, and all tunable parameters in one place
data_acquisition.py      USGS historical catalog fetch + real-time ATOM feed polling
feature_engineering.py   15 seismic indicator features + binary label construction
data_augmentation.py     CTGAN synthetic minority-class generation
models.py                StaticModel and AdaptiveModel (XGBoost + replay buffer)
spatial_predictor.py     Magnitude/recency-weighted spatial clustering → zone,
                         centroid, and uncertainty radius
prediction_engine.py     Turns model output + spatial stats into structured predictions
live_updater.py          LiveUpdater class used by `main.py --mode live`
live_demo.py             Standalone real-time terminal dashboard (continuous polling
                         + adaptive updates + prediction refresh)
generate.py              Lightweight refresh: loads the saved model and regenerates
                         predictions without retraining (~60s) — used by CI
main.py                  Master pipeline entry point (train/update/evaluate/live/predict)
index.html               Static front-end dashboard (map + predictions + live USGS feed)
.github/workflows/       GitHub Actions: refreshes predictions every 6 hours and
                         deploys them to the live site
```

Data flows: `data_acquisition.py` → `feature_engineering.py` → (`data_augmentation.py` →)
`models.py` → `spatial_predictor.py` / `prediction_engine.py` → `predictions/latest_predictions.json`
→ `index.html`.

## Getting started

Requires Python 3.10+ (developed and tested on 3.13).

```bash
git clone https://github.com/richiemin28/EarthQuakePredictor.git
cd EarthQuakePredictor
pip install -r requirements.txt
```

`requirements.txt` covers the full pipeline (including CTGAN/`sdv`, used only for
training). `requirements_server.txt` is the trimmed dependency set used by the
lightweight CI refresh job (`generate.py`).

### Run the full pipeline

```bash
# Fetch USGS data (1990-2025), engineer features, train both models,
# run the evaluation, and generate predictions:
python main.py --mode full
```

This takes a while on first run — the historical fetch and rolling-window feature
computation are the slow steps (tens of minutes). Results are cached to `data/` and
`models/` so subsequent runs are fast.

### Other modes

```bash
python main.py --mode train            # (Re)train static + adaptive models from scratch
python main.py --mode train --refresh  # Force a fresh USGS download instead of using the cache
python main.py --mode evaluate         # Pseudo-prospective evaluation, static vs adaptive
python main.py --mode predict          # Generate structured predictions from the saved model
python main.py --mode update           # Fetch new events since the last run, adapt the model
python main.py --mode live             # Start the continuous live-updating loop
```

### Live terminal dashboard

```bash
python live_demo.py                 # Poll USGS every 5 minutes (default)
python live_demo.py --interval 60   # Poll every 60 seconds
```

Polls the USGS ATOM feed, ingests any new events in the Myanmar bounding box, triggers an
adaptive model update, and redraws a full prediction dashboard (recent activity, zone
rankings, forward predictions with coordinates/radius/probability) in the terminal. Press
`Ctrl+C` to stop — the model checkpoints automatically on exit.

### Web dashboard

`index.html` is a self-contained static page — open it directly in a browser, or serve it
with any static file server. It reads `predictions/latest_predictions.json` (produced by
`generate.py` or `main.py --mode predict`) for the ML forecasts, and polls the USGS ATOM
feed directly in the browser for live activity.

The included GitHub Actions workflow (`.github/workflows/update-predictions.yml`) refreshes
predictions every 6 hours using the already-trained model (no retraining), commits the
updated JSON back to the repo, and optionally deploys it to a host over FTP if
`FTP_SERVER` / `FTP_USERNAME` / `FTP_PASSWORD` secrets are configured. If you fork this
repo without those secrets, the FTP step simply no-ops (`continue-on-error: true`) and the
predictions JSON still updates in-repo.

## Data sources

- **[USGS Earthquake Catalog API](https://earthquake.usgs.gov/fdsnws/event/1/)** —
  FDSN-compliant historical catalog, M2.0+, bounding box 10–30°N / 90–105°E, 1990–2025.
- **[USGS ATOM feed](https://earthquake.usgs.gov/earthquakes/feed/)** — real-time event
  stream, polled for live updates.

Both are free, public, and require no API key.

## Limitations

Carried over directly from the dissertation's own assessment (Chapter 6.2) — worth
reading before trusting any output:

- **Catalog sparsity.** Myanmar's public USGS catalog under-represents actual seismicity
  compared to well-monitored regions; a dedicated seismic array study (Yang et al., 2024)
  found roughly double the events of routine processing.
- **Reduced feature set.** 15 of the 61 features used by the reference study (Mukherjee
  et al., 2025) are implemented here, chosen by their reported SHAP importance.
- **Simple continual-learning strategy.** Fixed-ratio replay buffering is the simplest
  viable catastrophic-forgetting mitigation; it produces a non-monotonic
  year-by-year trajectory (a temporary dip in 2023) rather than uniformly increasing
  performance.
- **Location estimates are descriptive, not physical.** Coordinates and radii come from
  clustering of recent seismicity, not from fault-geometry or stress modelling — they show
  *where recent activity has concentrated*, not a physically derived rupture probability.
- **No operational deployment.** This has been validated in pseudo-prospective evaluation
  and confirmed to run correctly in live polling mode, but has not been reviewed by
  seismologists or integrated with any civil protection framework.

## Disclaimer

The Myanmar Earthquake Prediction System is an experimental research artefact developed
for academic purposes only. Its probabilistic predictions are based on statistical
pattern analysis of historical and real-time seismic catalog data and are **not
validated for operational use**. They do not constitute official earthquake warnings,
hazard assessments, or civil protection advisories of any kind. No reliance should be
placed on the system's outputs for emergency response, public safety, or infrastructure
planning without independent validation by qualified seismologists and integration with
established civil protection frameworks. Deterministic earthquake prediction (precise
time, location, and magnitude) remains beyond the reach of current science — this system
produces probabilistic forecasts within defined spatial, temporal, and magnitude windows,
nothing more.

## Citation

If you build on this work, please cite the dissertation:

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

Thanks to Dr Joe Collenette for supervision, and to the **United States Geological
Survey** for maintaining free, open access to the earthquake catalog API and real-time
ATOM feed that this entire project is built on. All research design, experimental
decisions, and analysis are the author's own.

## License

No license file is currently included in this repository — all rights are reserved by
default under copyright law. If you'd like to reuse, modify, or redistribute this code,
please open an issue or contact the author first.
