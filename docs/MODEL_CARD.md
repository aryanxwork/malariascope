# Model card: MalariaScope

*Fill in the bracketed values from `artifacts/latest/metadata.json` and `metrics.json` after training.*

## Model details
- **Task:** binary classification of a single segmented red-blood-cell image: *Parasitized* (positive) vs *Uninfected*.
- **Architecture:** EfficientNet-B0 (ImageNet-pretrained) + global average pooling + dropout + 1-logit head. Baseline: 8-conv VGG-style CNN trained from scratch. Edge variant: MobileNetV3-Small.
- **Input:** RGB, any size; letterboxed to 224×224 (EfficientNet-B0) / 128×128 (baseline, MobileNetV3), raw 0–255 pixels (scaling inside the model).
- **Output:** a logit → `sigmoid(logit / T)` with temperature T = [ ] fitted on validation.
- **Decision rule:** parasitized if p ≥ [threshold], chosen on validation patients for ≥98% sensitivity; |p − threshold| < 0.15 → "needs review".
- **Version / run:** [run id], trained [date], TensorFlow [version].

## Intended use
- Research, teaching and portfolio demonstration of trustworthy medical-image ML practice.
- **Not intended** for clinical diagnosis, treatment decisions, or use without a qualified microscopist.

## Training data
- NIH/NLM malaria cell dataset: 27,558 cell crops (13,779 per class) from Giemsa-stained thin smears of 150 *P. falciparum*-infected and 50 healthy patients, Chittagong Medical College Hospital, Bangladesh.
- Split by **patient** (StratifiedGroupKFold, 7 folds → ~72% train / 14% validation / 14% test), zero patient overlap asserted. Split file saved with every run.

## Evaluation (held-out patients)
| Metric | Value | 95% CI (patient bootstrap) |
|---|---|---|
| Accuracy | [ ] | [ ] |
| Sensitivity | [ ] | [ ] |
| Specificity | [ ] | [ ] |
| F1 | [ ] | [ ] |
| ROC-AUC | [ ] | [ ] |
| ECE (raw → calibrated) | [ ] → [ ] | |
| Cell-split ablation accuracy (leaky) | [ ] | |

## Known limitations and risks
- Single site, single species, pre-segmented cells; distribution shift (new microscope, stain protocol, camera) is likely to degrade performance silently.
- Dataset contains some label noise.
- Grad-CAM is a coarse (feature-map resolution) post-hoc explanation and can look plausible even when the model is wrong.
- Out-of-distribution detection is a simple heuristic (dark-background check), not a learned detector.

## Ethical considerations
- A false negative (missed infection) can delay treatment. The operating point favours sensitivity, and borderline cases are routed to human review.
- Any real deployment would require prospective validation on local data, regulatory approval and clinician oversight.
