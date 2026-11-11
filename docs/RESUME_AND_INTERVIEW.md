# Resume entry and interview prep

## Resume entry (fill the [ ] only with numbers your own run printed)

**MalariaScope: Explainable Malaria Detection from Blood-Smear Images** | Python, TensorFlow/Keras, scikit-learn, FastAPI, React, TypeScript, Tailwind, Docker | [github.com/aryanxwork/malariascope]

- Built an end-to-end malaria screening system on the NIH thin-smear dataset (27,558 cells, 200 patients). Split train/test **by patient** to remove data leakage; a cell-level-split ablation showed leakage inflating accuracy by [X] points.
- Fine-tuned EfficientNet-B0 with two-phase transfer learning: [X]% sensitivity, [X]% specificity, [X] ROC-AUC on **unseen patients** (95% CIs via patient-level bootstrap), vs [X]% accuracy for a from-scratch CNN baseline.
- Calibrated outputs with temperature scaling (ECE [X] → [X]), tuned the decision threshold for ≥98% sensitivity, and added Grad-CAM heatmaps plus a needs-review band for borderline cells.
- Shipped a FastAPI + React/Tailwind app (batch upload, heatmaps, parasitemia estimate, metrics dashboard); exported an int8 TFLite model ([X] MB, [X] ms/cell on CPU); Docker Compose, 27 automated tests, GitHub Actions CI.

**If you only have room for 3 bullets:** merge bullets 2 and 3.

Notes:
- Your original bullet said "validation accuracy" with no number. Report **test** metrics on held-out patients, not validation accuracy. Validation data was used for early stopping, calibration and threshold choice, so it is not an unbiased estimate.
- If the patient-split number comes out lower than the "97–99%" you see in blog posts, that's expected. The ablation row explains why, and it's your strongest talking point.

## Interview questions you should be able to answer

1. **Why split by patient?** Each patient contributes 100+ correlated cells. A random split tests the model on patients it has already seen, which inflates accuracy. The dataset authors report 98.61% (cell split) vs 95.90% (patient split). I measured my own gap with the ablation config.
2. **Why is sensitivity prioritised over accuracy?** In screening, a false negative means an untreated infection, while a false positive means a microscopist double-checks. So I pick the threshold that guarantees ≥98% validation sensitivity, then maximise specificity.
3. **What does temperature scaling do, and why doesn't it change AUC?** It divides logits by one scalar T fitted by NLL on validation. That's a monotonic transform, so the ranking (and therefore ROC-AUC) is unchanged, but probabilities now match observed frequencies (lower ECE).
4. **Why bootstrap by patient instead of by cell?** Cells from one patient aren't independent. Resampling cells gives confidence intervals that are too narrow.
5. **Two-phase fine-tuning, and why keep BatchNorm frozen?** Phase 1 trains only the new head, so random gradients don't wreck the pretrained features. Phase 2 unfreezes the top layers at a ~100× lower learning rate. BN stays in inference mode because re-estimating its statistics on small batches of a very different domain destabilises training.
6. **Grad-CAM limitations?** It's coarse (feature-map resolution, 7×7 at 224 px), post hoc, and can look convincing even when the model is wrong. It's a sanity check that the model attends to the parasite and not to stain artefacts, not proof of correctness.
7. **How do you prevent train/serve skew?** The API imports the exact preprocessing function used in training, and pixel scaling lives inside the model graph.
8. **What happens if someone uploads a random photo?** The model will still output a number. The API runs input checks (dark background, image size), flags the image, the UI marks it unreliable, and it's excluded from the parasitemia estimate. A learned OOD detector would be the proper fix.
9. **What would you do next?** Detect and segment cells on full field-of-view images (e.g. a YOLO/U-Net stage) so the system works on raw smear photos; validate on a second site (e.g. NIH's thick-smear or P. vivax sets); learned OOD detection; species and life-stage classification.
10. **Why EfficientNet-B0 over ResNet50/VGG?** It's ~6× fewer parameters than ResNet50 with comparable ImageNet accuracy, which matters for the edge/TFLite target. Include your own baseline-vs-transfer numbers when you answer.
