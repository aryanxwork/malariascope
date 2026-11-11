const STEPS = [
  {
    title: "1 · Leakage-free data split",
    body: "27,558 cell crops come from ~200 patients. Cells from one patient look alike (same slide, stain and microscope), so a random cell-level split lets the model 'recognise the slide'. MalariaScope splits by patient (StratifiedGroupKFold) and asserts zero overlap. A cell-split ablation is included to measure the inflation.",
  },
  {
    title: "2 · Preprocessing shared by training and serving",
    body: "Aspect-preserving resize with black letterboxing (matches the dataset background). Pixel scaling lives inside the model graph, and the API imports the exact same preprocessing function, so there is no train/serve skew.",
  },
  {
    title: "3 · Augmentation for real-world variation",
    body: "Full rotations and flips (cells have no orientation) plus stain-colour jitter in hue, saturation, brightness and contrast to mimic different Giemsa batches and microscopes.",
  },
  {
    title: "4 · Models",
    body: "A from-scratch CNN baseline, then ImageNet-pretrained EfficientNet-B0 with two-phase transfer learning (frozen backbone, then top layers fine-tuned at a low learning rate with BatchNorm frozen). MobileNetV3-Small is available for edge deployment and exports to TFLite.",
  },
  {
    title: "5 · Calibration and a clinical operating point",
    body: "Temperature scaling fitted on validation makes '90% parasitized' mean 90%. The decision threshold is set on validation for ≥98% sensitivity, because missing an infection is worse than a false alarm. Borderline cells are flagged for human review instead of being forced into a label.",
  },
  {
    title: "6 · Honest evaluation and explainability",
    body: "Accuracy, sensitivity, specificity, F1, MCC, ROC-AUC and PR-AUC on held-out patients, with patient-level bootstrap confidence intervals. Grad-CAM shows which pixels drove the 'parasitized' score, so you can check that the model looks at the parasite and not at staining artefacts.",
  },
];

export default function AboutPage() {
  return (
    <div className="max-w-4xl space-y-6">
      <section>
        <h1 className="text-2xl font-semibold tracking-tight">How MalariaScope works</h1>
        <p className="mt-2 text-sm text-ink-2">
          Microscopy of Giemsa-stained thin blood smears is the standard way to confirm malaria and count parasitemia, but it depends on a trained
          microscopist. MalariaScope classifies single red-blood-cell crops as <b>parasitized</b> or <b>uninfected</b> and is built to be
          honest about how well it works.
        </p>
      </section>
      <ol className="grid gap-3 md:grid-cols-2">
        {STEPS.map((s) => (
          <li key={s.title} className="rounded-xl border border-line bg-panel p-4">
            <p className="text-sm font-semibold">{s.title}</p>
            <p className="mt-1.5 text-sm leading-relaxed text-ink-2">{s.body}</p>
          </li>
        ))}
      </ol>
      <section className="rounded-xl border border-amber-300 bg-review-bg p-4 text-sm text-review">
        <p className="font-semibold">Limitations</p>
        <ul className="mt-1.5 list-disc space-y-1 pl-5">
          <li>Trained on one dataset from one hospital (Chittagong, Bangladesh), P. falciparum only, with pre-segmented cells. Performance on other sites, species or stains is unknown.</li>
          <li>It classifies single cells. It does not segment whole fields of view, detect species or stage, or diagnose a patient.</li>
          <li>The input checks only catch obvious out-of-distribution images. A confident score on an unusual image means nothing.</li>
          <li>Research prototype, not a medical device.</li>
        </ul>
      </section>
    </div>
  );
}
