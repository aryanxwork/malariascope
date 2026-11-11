import { useEffect, useState } from "react";
import { CheckCircle2, Loader2 } from "lucide-react";
import { api, pct, type BinaryMetrics, type ModelInfo } from "../lib/api";
import { ChartCard, CurveChart, SERIES } from "../components/Charts";

export default function PerformancePage() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.model().then(setInfo).catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <p className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">Could not load model info: {error}</p>;
  if (!info) return <p className="flex items-center gap-2 text-sm text-ink-3"><Loader2 size={16} className="animate-spin" /> Loading model report…</p>;
  const m = info.metrics;
  if (!m) return <p className="text-sm text-ink-3">This model has no metrics.json. Re-run evaluation.</p>;

  const t = m.test;
  const a = t.at_threshold;
  const ci = t.ci95_patient_bootstrap;
  const leak = Object.values(m.splits.leakage).reduce((s, v) => s + v, 0);
  const hist = m.history;
  const epochs = (key: string) => (hist[key] ?? []).map((y, i) => ({ x: i + 1, y }));

  return (
    <div className="space-y-6">
      <section className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Model performance</h1>
          <p className="mt-1 text-sm text-ink-2">
            <span className="font-mono text-xs">{info.metadata.model}</span> · run <span className="font-mono text-xs">{info.metadata.run}</span> ·{" "}
            {m.params.total ? `${(m.params.total / 1e6).toFixed(2)}M params` : ""} · input {m.img_size}×{m.img_size}
          </p>
        </div>
        <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs ${m.split_mode === "patient" && leak === 0 ? "border-emerald-200 bg-emerald-50 text-emerald-800" : "border-amber-300 bg-review-bg text-review"}`}>
          <CheckCircle2 size={14} />
          {m.split_mode === "patient" ? `Patient-level split · ${leak} patients shared between splits` : "Cell-level split (leaky ablation)"}
        </span>
      </section>

      <p className="text-sm text-ink-2">
        All numbers below are on <b>{t.n.toLocaleString()} held-out test cells from {m.splits.test.patients} patients the model never saw</b>, at the operating
        threshold chosen on validation data. Brackets are 95% confidence intervals from a patient-level bootstrap.
      </p>

      <section className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
        <Tile label="Accuracy" v={a.accuracy} ci={ci.accuracy} />
        <Tile label="Sensitivity (recall)" v={a.sensitivity} ci={ci.sensitivity} />
        <Tile label="Specificity" v={a.specificity} ci={ci.specificity} />
        <Tile label="F1 score" v={a.f1} ci={ci.f1} />
        <Tile label="ROC-AUC" v={t.roc_auc} ci={ci.roc_auc} raw />
        <Tile label="PR-AUC" v={t.pr_auc} raw />
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <ChartCard title="ROC curve" subtitle={`AUC ${t.roc_auc.toFixed(3)} · orange dot = deployed operating point`}>
          <CurveChart
            series={[{ name: "Model", color: SERIES.primary, data: t.roc_curve.map((d) => ({ x: d.fpr, y: d.tpr })) }]}
            xLabel="False positive rate"
            yLabel="Sensitivity"
            diagonal
            point={{ x: 1 - a.specificity, y: a.sensitivity, label: `t=${a.threshold.toFixed(2)}` }}
          />
        </ChartCard>
        <ChartCard title="Precision–recall curve" subtitle={`Average precision ${t.pr_auc.toFixed(3)}`}>
          <CurveChart
            series={[{ name: "Model", color: SERIES.primary, data: t.pr_curve.map((d) => ({ x: d.recall, y: d.precision })) }]}
            xLabel="Recall"
            yLabel="Precision"
          />
        </ChartCard>
        <ConfusionMatrix m={a} />
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <ChartCard
          title="Calibration (reliability diagram)"
          subtitle={`Temperature scaling T=${m.calibration.temperature.toFixed(2)} · ECE ${m.calibration.ece_before.toFixed(3)} → ${m.calibration.ece_after.toFixed(3)}`}
        >
          <CurveChart
            diagonal
            series={[
              { name: "Raw", color: SERIES.secondary, dots: true, data: m.calibration.reliability_before.map((b) => ({ x: b.confidence, y: b.frequency })) },
              { name: "Calibrated", color: SERIES.primary, dots: true, data: m.calibration.reliability_after.map((b) => ({ x: b.confidence, y: b.frequency })) },
            ]}
            xLabel="Predicted probability"
            yLabel="Observed frequency"
          />
        </ChartCard>
        {hist.loss?.length ? (
          <ChartCard title="Training loss" subtitle={m.finetune_start_epoch ? "Vertical line: backbone fine-tuning starts" : "Binary cross-entropy"}>
            <CurveChart
              integerX
              xDomain={[1, "auto"]}
              yDomain={[0, "auto"]}
              series={[
                { name: "Train", color: SERIES.primary, data: epochs("loss") },
                { name: "Validation", color: SERIES.secondary, data: epochs("val_loss") },
              ]}
              verticalAt={m.finetune_start_epoch ? { x: m.finetune_start_epoch + 0.5, label: "fine-tune" } : undefined}
              xLabel="Epoch"
              yLabel="Loss"
            />
          </ChartCard>
        ) : null}
        {hist.auc?.length ? (
          <ChartCard title="Training ROC-AUC" subtitle="Validation AUC drives early stopping">
            <CurveChart
              integerX
              xDomain={[1, "auto"]}
              yDomain={["auto", 1]}
              series={[
                { name: "Train", color: SERIES.primary, data: epochs("auc") },
                { name: "Validation", color: SERIES.secondary, data: epochs("val_auc") },
              ]}
              verticalAt={m.finetune_start_epoch ? { x: m.finetune_start_epoch + 0.5, label: "fine-tune" } : undefined}
              xLabel="Epoch"
              yLabel="AUC"
            />
          </ChartCard>
        ) : null}
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-line bg-panel p-4">
          <p className="text-sm font-semibold">Decision threshold</p>
          <p className="mt-1 text-xs text-ink-3">
            For screening, a missed infection costs more than a false alarm. The threshold is chosen on <i>validation</i> patients to keep at least{" "}
            {pct(m.threshold.target_sensitivity, 0)} sensitivity with the best possible specificity, placed mid-way in the safe range rather than on its edge; cells within ±{m.threshold.review_margin} of it are flagged for human review
            ({pct(t.review_rate)} of test cells{t.accuracy_excluding_review != null ? `; accuracy on the rest ${pct(t.accuracy_excluding_review)}` : ""}).
          </p>
          <table className="tabular mt-3 w-full text-sm">
            <thead className="text-left text-xs text-ink-3">
              <tr className="border-b border-line">
                <th className="py-1.5 font-normal">Threshold</th>
                <th className="font-normal">Sensitivity</th>
                <th className="font-normal">Specificity</th>
                <th className="font-normal">F1</th>
                <th className="font-normal">Missed infections</th>
              </tr>
            </thead>
            <tbody>
              {[["Naive 0.50", t.at_0_5], [`Tuned ${a.threshold.toFixed(3)}`, a]].map(([name, r]) => {
                const x = r as BinaryMetrics;
                return (
                  <tr key={name as string} className="border-b border-line last:border-0">
                    <td className="py-1.5">{name as string}</td>
                    <td>{pct(x.sensitivity)}</td>
                    <td>{pct(x.specificity)}</td>
                    <td>{x.f1.toFixed(3)}</td>
                    <td>{x.confusion.fn.toLocaleString()}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="rounded-xl border border-line bg-panel p-4">
          <p className="text-sm font-semibold">Data splits</p>
          <p className="mt-1 text-xs text-ink-3">Grouped by patient ID parsed from NIH file names, stratified by label.</p>
          <table className="tabular mt-3 w-full text-sm">
            <thead className="text-left text-xs text-ink-3">
              <tr className="border-b border-line">
                <th className="py-1.5 font-normal">Split</th>
                <th className="font-normal">Patients</th>
                <th className="font-normal">Cells</th>
                <th className="font-normal">Parasitized</th>
                <th className="font-normal">Uninfected</th>
              </tr>
            </thead>
            <tbody>
              {(["train", "val", "test"] as const).map((k) => (
                <tr key={k} className="border-b border-line last:border-0">
                  <td className="py-1.5 capitalize">{k === "val" ? "validation" : k}</td>
                  <td>{m.splits[k].patients}</td>
                  <td>{m.splits[k].cells.toLocaleString()}</td>
                  <td>{m.splits[k].parasitized.toLocaleString()}</td>
                  <td>{m.splits[k].uninfected.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function Tile({ label, v, ci, raw }: { label: string; v: number; ci?: [number, number]; raw?: boolean }) {
  const f = (x: number) => (raw ? x.toFixed(3) : pct(x));
  return (
    <div className="rounded-xl border border-line bg-panel p-4">
      <p className="text-xs text-ink-3">{label}</p>
      <p className="tabular mt-1 text-2xl font-semibold tracking-tight">{f(v)}</p>
      <p className="tabular mt-0.5 text-[11px] text-ink-3">{ci ? `95% CI ${f(ci[0])}–${f(ci[1])}` : " "}</p>
    </div>
  );
}

function ConfusionMatrix({ m }: { m: BinaryMetrics }) {
  const { tp, fp, tn, fn } = m.confusion;
  const rows = [
    { actual: "Uninfected", cells: [{ v: tn, ok: true, tag: "TN" }, { v: fp, ok: false, tag: "FP" }] },
    { actual: "Parasitized", cells: [{ v: fn, ok: false, tag: "FN" }, { v: tp, ok: true, tag: "TP" }] },
  ];
  return (
    <figure className="rounded-xl border border-line bg-panel p-4">
      <figcaption className="mb-3">
        <p className="text-sm font-semibold">Confusion matrix</p>
        <p className="mt-0.5 text-xs text-ink-3">Row-normalised; shading = share of the actual class</p>
      </figcaption>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-[2px] text-sm">
        <span />
        <span className="pb-1 text-center text-xs text-ink-3">Pred. uninfected</span>
        <span className="pb-1 text-center text-xs text-ink-3">Pred. parasitized</span>
        {rows.map((r) => {
          const total = r.cells[0].v + r.cells[1].v || 1;
          return [
            <span key={`${r.actual}-l`} className="flex items-center pr-2 text-xs text-ink-3">Actual {r.actual.toLowerCase()}</span>,
            ...r.cells.map((c) => {
              const share = c.v / total;
              const dark = share > 0.55;
              return (
                <div
                  key={`${r.actual}-${c.tag}`}
                  title={`${c.tag}: ${c.v.toLocaleString()} (${pct(share)})`}
                  className="tabular flex h-24 flex-col items-center justify-center rounded-md"
                  style={{ background: `rgba(42,120,214,${0.08 + 0.85 * share})`, color: dark ? "#fff" : "#0b0b0b" }}
                >
                  <span className="text-xl font-semibold">{c.v.toLocaleString()}</span>
                  <span className="text-xs opacity-80">{c.tag} · {pct(share)}</span>
                </div>
              );
            }),
          ];
        })}
      </div>
    </figure>
  );
}
