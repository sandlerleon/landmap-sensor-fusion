# Uncertainty-aware multimodal sensor fusion for landfill mining

Leon Sandler, Independent Researcher — sandler.leon@gmail.com
ORCID [0009-0007-4584-808X](https://orcid.org/0009-0007-4584-808X)

Models, validation, economic analysis and figures for *"Uncertainty-Aware Multimodal
Sensor Fusion for Economic Prioritization of Landfill Mining: A Simulation-Based
Evaluation,"* submitted to **Environmental Monitoring and Assessment**.

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
landfill-mining excavation decision, and by how much.

## What the models say

| Question | Result |
|---|---|
| Per-class F1, best classifier (XGBoost), spatial holdout | metal **0.99**, organic **0.98**, C&D **0.87**; plastic **0.32**, glass **0.24** |
| Random vs. spatial-block validation gap | small, 95% CI includes zero for all three classifiers — a tested null result |
| Distance-to-nearest-training-cell decay | flat (~0.70 accuracy at 1 m to 10+ m) |
| Sensor most important for classification | GPR (macro-F1 0.655 → 0.599 when removed) |
| Sensor with reliably positive economic value (VOI) | **only the magnetometer** (positive in 100% of surveys); GPR/EMI/thermal/gas are not |
| NIR augmentation | plastic F1 0.32 → 0.38; VOI unchanged (statistically) |
| Gas-hazard classifier | ROC-AUC 0.94, 100% recall at a safety-oriented operating point |
| VOI under baseline assumptions | median **$663** (averaged over 20 surveys), positive in 95% |
| Break-even survey cost | median **$1,451** |
| Synthetic-to-real degradation: VOI = 0 boundary | 93% of the way from baseline to the most severe combined scenario tested |

## Contents

```
code/
  landmap_synth.py          synthetic latent-material field + 5 sensor-response channels
                             (+ optional 6th, near-infrared)
  landmap_model.py           spatial vs. random validation, 3 classifiers, calibration,
                             ablation, NIR augmentation, distance-decay diagnostic
  landmap_econ.py             Resource Value Score, Value of Information, break-even
                             survey cost, per-sensor economic value attribution
  landmap_robustness.py      synthetic-to-real degradation sweep and VOI=0 falsification
                             boundary search
  harvest_refs.py             Crossref metadata + abstracts for every DOI reference
  make_figures.py             Figures 1-9 (300 dpi PNG + 600 dpi TIFF)
  build_manuscript.py         manuscript; every number read from the JSON outputs
  build_cover_letter.py       cover letter
  audit_manuscript.py         checks the document against the model and against itself
figures/                      Figures 1-9
manuscript/                   manuscript and cover letter
```

## Reproducing

```bash
pip install -r requirements.txt
cd code
python landmap_synth.py          # writes landmap_synth_config.json (assumptions dump)
python landmap_model.py          # -> landmap_model_results.json (seed 20260926, 40 repeats)
python landmap_econ.py           # -> landmap_econ_results.json (20 repeats x 20,000 MC draws)
python landmap_robustness.py     # -> landmap_robustness_results.json
python harvest_refs.py           # -> _refs.json, _abstracts.txt
python make_figures.py           # -> ../figures/
python build_manuscript.py
python build_cover_letter.py
python audit_manuscript.py
```

## Key methodological notes

- The classification baseline uses **five** material-response channels (magnetometer,
  EMI, GPR, thermal, gas); LiDAR/RGB surface topography provides georeferencing and
  cell-depth context rather than an independent classification input, and a **sixth**,
  near-infrared/optical channel is evaluated separately as a proposed augmentation.
- Spatial-vs-random validation and a distance-to-nearest-training-cell diagnostic were
  both tested honestly; neither showed the leakage effect expected from prior spatial-CV
  literature in this particular simulated configuration, and this null result is reported
  rather than a manufactured positive one.
- Classification-utility (F1) and economic-utility (VOI) sensor rankings diverge: GPR
  matters most for classification, but only the magnetometer reliably moves the VOI in
  this economic model. This divergence is the paper's central novel finding.
- One DOI in the original industrial proposal this work is based on (Laner et al. 2019)
  was verified via Crossref to be incorrect, resolving to an unrelated paper; the correct
  DOI is used throughout (`10.1016/j.wasman.2019.07.007`).

## Citation

Code and data archived at Zenodo (DOI to be added on release); manuscript preprint DOI to
be added if/when deposited.

## License

Code MIT (`LICENSE`); manuscript text and figures CC BY 4.0.
