# From classification accuracy to decision value: sensor fusion for landfill mining

Leon Sandler, Independent Researcher — sandler.leon@gmail.com
ORCID [0009-0007-4584-808X](https://orcid.org/0009-0007-4584-808X)

Models, validation, economic analysis and figures for *"From Classification Accuracy to
Decision Value: Multimodal Sensor Fusion for Economic Prioritization of Landfill Mining
— A Simulation-Based Value-of-Information Analysis,"* submitted to **Environmental
Monitoring and Assessment**.

> **Every performance figure here is a model prediction from a synthetic, physically
> motivated but unvalidated simulation.** No field, laboratory or test-bed data were
> used. Sensor-response coefficients, noise levels and every economic parameter are
> stated assumptions, labelled as such in the code, and swept rather than fixed.

## The question

An architecture for pre-excavation landfill characterisation — a magnetometer,
electromagnetic induction, ground-penetrating radar, thermal imaging and a gas array,
fused by machine learning into material and hazard probabilities and coupled to an
economic value-of-information layer — was first proposed as an industrial open-innovation
submission. This repository tests, by simulation, whether it would actually change a
landfill-mining excavation decision, whether the right sensor is a fixed property of the
architecture, and by how much.

## What the models say

| Question | Result |
|---|---|
| Per-class F1, best classifier (XGBoost), spatial holdout | metal **0.99**, organic **0.98**, C&D **0.87**; plastic **0.32**, glass **0.24** |
| Random vs. spatial-block validation gap | small, 95% CI includes zero for all three classifiers — a tested null result |
| Sensor most important for classification | GPR (macro-F1 0.655 → 0.599 when removed) |
| Sensor with reliably positive economic value at baseline economics | **only the magnetometer** (positive in 100% of surveys) |
| Does that ranking survive ordinary price fluctuation? | Yes — a 5×5 grid over metal/plastic price never changed the winner |
| Does it survive a different site composition? | **No** — polymer-rich, C&D-rich and glass-rich archetypes shift the top sensor to GPR or EMI |
| Does the economic-layer classifier choice (RF vs. XGBoost, raw or calibrated) matter? | No — VOI differs by <2% across all three |
| NIR augmentation (hypothetical response coefficients) | plastic F1 0.32 → 0.38; VOI unchanged (statistically) |
| Gas-hazard classifier | ROC-AUC 0.94, 100% recall, but only 52% precision at the safety operating point |
| VOI under baseline assumptions | median **$924** (pooled over 40 independent surveys, two batches), positive in 92.5% |
| Break-even survey cost | median **$1,451** |
| 2-D decision-viability boundary | far more sensitive to economic adversity (β) than to sensor degradation (α) |

## Contents

```
code/
  landmap_synth.py             synthetic latent-material field + 5 material-response
                                channels (+ optional 6th, near-infrared)
  landmap_model.py              spatial vs. random validation, 3 classifiers, calibration,
                                ablation, NIR augmentation, distance-decay diagnostic
  landmap_econ.py                Resource Value Score, Value of Information, break-even
                                survey cost, per-sensor economic value attribution
  landmap_econ_fast.py           vectorized VOI Monte Carlo used by the two map scripts
  landmap_robustness.py         synthetic-to-real degradation sweep and 1-D VOI-zero
                                boundary search
  landmap_hazard_extra.py        gas-hazard precision/specificity/PR-AUC (recall alone
                                overstates the operating point)
  landmap_calib_voi.py           does the economic-layer classifier choice matter? RF vs.
                                raw/Platt-calibrated XGBoost, same VOI protocol
  landmap_sensor_map.py          sensor-selection grid over metal/plastic price multipliers
  landmap_sensor_map_extreme.py  sensor-selection under deliberately extreme site archetypes
  landmap_voi_map.py             2-D decision-viability map, VOI(alpha, beta)
  landmap_baseline_reconcile.py  pools the economic-layer and robustness-sweep baseline
                                VOI batches into one canonical estimate with a proper SE
  harvest_refs.py                Crossref metadata + abstracts for every DOI reference
  make_figures.py                Figures 1-11 (300 dpi PNG + 600 dpi TIFF)
  build_manuscript.py            manuscript; every number read from the JSON outputs
  build_cover_letter.py          cover letter
  audit_manuscript.py            checks the document against the model and against itself
figures/                         Figures 1-11
manuscript/                      manuscript and cover letter
```

## Reproducing

```bash
pip install -r requirements.txt
cd code
python landmap_synth.py            # writes landmap_synth_config.json (assumptions dump)
python landmap_model.py            # -> landmap_model_results.json (seed 20260926, 40 repeats)
python landmap_econ.py             # -> landmap_econ_results.json (20 repeats x 20,000 MC draws)
python landmap_robustness.py       # -> landmap_robustness_results.json
python landmap_hazard_extra.py     # -> landmap_hazard_extra_results.json
python landmap_calib_voi.py        # -> landmap_calib_voi_results.json
python landmap_sensor_map.py       # -> landmap_sensor_map_results.json
python landmap_sensor_map_extreme.py  # -> landmap_sensor_map_extreme_results.json
python landmap_voi_map.py          # -> landmap_voi_map_results.json
python landmap_baseline_reconcile.py  # -> landmap_baseline_reconcile_results.json
python harvest_refs.py             # -> _refs.json, _abstracts.txt
python make_figures.py             # -> ../figures/
python build_manuscript.py
python build_cover_letter.py
python audit_manuscript.py
```

## Key methodological notes

- The classification baseline uses **five** material-response channels (magnetometer,
  EMI, GPR, thermal, gas); LiDAR/RGB surface topography provides georeferencing and
  cell-depth context rather than an independent classification input, and a **sixth**,
  near-infrared/optical channel is evaluated separately as a hypothetical augmentation
  under assumed response coefficients, not measured spectra.
- Spatial-vs-random validation and a distance-to-nearest-training-cell diagnostic were
  both tested honestly; neither showed the leakage effect expected from prior spatial-CV
  literature in this particular simulated configuration, and this null result is reported
  rather than a manufactured positive one.
- **The paper's central finding**: classification-utility (F1) and economic-utility (VOI)
  sensor rankings diverge, and the economic ranking is itself conditional. GPR matters
  most for classification; the magnetometer is the reliable economic winner at baseline
  economics; but a deliberately extreme site archetype (polymer-, C&D-, or glass-rich)
  flips the economic winner to GPR or EMI. Optimal sensor selection is a function of site
  composition, market economics and sensor uncertainty jointly, not a fixed property of
  the sensing architecture.
- The economic layer uses random forest rather than XGBoost (the best raw classifier)
  because XGBoost's probabilities are less well calibrated; `landmap_calib_voi.py` tests
  this choice directly and finds it does not change the VOI conclusion.
- The economic layer's headline VOI batch (20 surveys) and the robustness sweep's own
  undegraded reference point (a second, matched-protocol batch) are independent Monte
  Carlo estimates of the same baseline quantity and differ by 1.4 SE of their difference —
  unsurprising sampling noise for a 20-survey average of a right-skewed distribution, not a
  methodology inconsistency. `landmap_baseline_reconcile.py` pools both into one 40-survey
  canonical estimate rather than reporting two numbers side by side.
- The 1-D combined-degradation VOI-zero boundary is extended to a full 2-D
  decision-viability map (`landmap_voi_map.py`) over sensing-degradation severity and an
  independent economic-adversity severity axis; the boundary is far more sensitive to the
  latter.
- One DOI in the original industrial proposal this work is based on (Laner et al. 2019)
  was verified via Crossref to be incorrect, resolving to an unrelated paper; the correct
  DOI is used throughout (`10.1016/j.wasman.2019.07.007`).

## Citation

- Code and data: [10.5281/zenodo.22984668](https://doi.org/10.5281/zenodo.22984668) (release v1.0.0)
- Manuscript preprint: [10.5281/zenodo.22984669](https://doi.org/10.5281/zenodo.22984669) (concept DOI,
  always resolves to the latest version; current: v1.0.1, a copyedited manuscript with several
  sense-changing errors introduced by an external copyediting pass corrected — see
  `manuscript_repair/fix_rubriq_v2.py` for the exact list. Results, figures, tables and
  references are unchanged from v1.0.0.)
- A line-number-free version for engrXiv submission is at `manuscript/LANDMAP_EMA_Manuscript_engrXiv.pdf`.

## License

Code MIT (`LICENSE`); manuscript text and figures CC BY 4.0.
