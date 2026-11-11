import { AlertTriangle, Bug, ShieldCheck, Flame } from "lucide-react";
import type { Prediction } from "../lib/api";
import { pct } from "../lib/api";

interface Props {
  name: string;
  src: string;
  result?: Prediction;
  error?: string;
  showHeatmap: boolean;
  onToggleHeatmap: () => void;
}

export default function ResultCard({ name, src, result, error, showHeatmap, onToggleHeatmap }: Props) {
  const heat = showHeatmap && result?.gradcam_png;
  const positive = result?.label === "Parasitized";
  const unreliable = !!result?.warnings.length;

  return (
    <article className="group flex flex-col overflow-hidden rounded-xl border border-line bg-panel">
      <div className="checker relative aspect-square">
        <img
          src={heat ? result!.gradcam_png! : src}
          alt={heat ? `Grad-CAM heatmap for ${name}` : name}
          className="size-full object-contain [image-rendering:auto]"
        />
        {result?.gradcam_png && (
          <button
            type="button"
            onClick={onToggleHeatmap}
            className="absolute right-2 top-2 inline-flex items-center gap-1 rounded-md bg-black/60 px-2 py-1 text-[11px] text-white opacity-90 hover:opacity-100"
            aria-pressed={!!heat}
          >
            <Flame size={12} /> {heat ? "Heatmap" : "Original"}
          </button>
        )}
        {result?.needs_review && (
          <span className="absolute left-2 top-2 inline-flex items-center gap-1 rounded-md bg-review-bg px-2 py-1 text-[11px] font-medium text-review">
            <AlertTriangle size={12} /> Needs review
          </span>
        )}
      </div>

      <div className="flex flex-1 flex-col gap-2 p-3">
        <p className="truncate text-xs text-ink-3" title={name}>
          {name}
        </p>
        {error && <p className="text-sm text-red-700">{error}</p>}
        {!result && !error && <div className="h-10 animate-pulse rounded bg-black/5" />}
        {result && (
          <>
            <div className={`flex items-center justify-between gap-2 ${unreliable ? "opacity-45" : ""}`}>
              <span className="inline-flex items-center gap-1.5 text-sm font-semibold">
                <span
                  className="grid size-5 place-items-center rounded-full text-white"
                  style={{ background: positive ? "var(--color-parasitized)" : "var(--color-uninfected)" }}
                >
                  {positive ? <Bug size={12} /> : <ShieldCheck size={12} />}
                </span>
                {result.label}
              </span>
              <span className="tabular text-sm text-ink-2">{pct(result.confidence)}</span>
            </div>
            <div className={unreliable ? "opacity-45" : ""}>
              <ProbabilityBar p={result.probability_parasitized} threshold={result.threshold} />
            </div>
            {unreliable && <p className="text-[11px] font-semibold text-review">Unreliable input — score not meaningful</p>}
            {result.warnings.map((w) => (
              <p key={w} className="flex gap-1 text-[11px] leading-snug text-review">
                <AlertTriangle size={12} className="mt-px shrink-0" /> {w}
              </p>
            ))}
          </>
        )}
      </div>
    </article>
  );
}

function ProbabilityBar({ p, threshold }: { p: number; threshold: number }) {
  const positive = p >= threshold;
  return (
    <div>
      <div className="relative h-1.5 rounded-full bg-black/[0.07]" role="img" aria-label={`P(parasitized) ${pct(p)}, threshold ${pct(threshold)}`}>
        <div
          className="h-full rounded-full"
          style={{ width: `${Math.max(p * 100, 1.5)}%`, background: positive ? "var(--color-parasitized)" : "var(--color-uninfected)" }}
        />
        <div className="absolute -top-1 h-3.5 w-0.5 rounded bg-ink" style={{ left: `calc(${threshold * 100}% - 1px)` }} />
      </div>
      <div className="tabular mt-1 flex justify-between text-[11px] text-ink-3">
        <span>P(parasitized) {pct(p)}</span>
        <span>threshold {threshold.toFixed(2)}</span>
      </div>
    </div>
  );
}
