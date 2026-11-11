import { useEffect, useMemo, useRef, useState } from "react";
import { Download, Flame, Loader2, Play, Trash2 } from "lucide-react";
import Dropzone from "../components/Dropzone";
import ResultCard from "../components/ResultCard";
import { api, type Prediction } from "../lib/api";

interface Item {
  id: string;
  file: File;
  url: string;
  result?: Prediction;
  error?: string;
}

type Filter = "all" | "Parasitized" | "Uninfected" | "review" | "warnings";
const CHUNK = 32;

export default function AnalyzePage({ modelReady }: { modelReady: boolean }) {
  const [items, setItems] = useState<Item[]>([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [gradcam, setGradcam] = useState(true);
  const [heatAll, setHeatAll] = useState(false);
  const [heatOverride, setHeatOverride] = useState<Record<string, boolean>>({});
  const [filter, setFilter] = useState<Filter>("all");
  const [sortByProb, setSortByProb] = useState(false);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => () => items.forEach((i) => URL.revokeObjectURL(i.url)), []); // eslint-disable-line react-hooks/exhaustive-deps

  const addFiles = (files: File[]) =>
    setItems((prev) => [
      ...prev,
      ...files.map((file) => ({ id: crypto.randomUUID(), file, url: URL.createObjectURL(file) })),
    ]);

  const clear = () => {
    abort.current?.abort();
    items.forEach((i) => URL.revokeObjectURL(i.url));
    setItems([]);
    setError(null);
    setHeatOverride({});
  };

  const run = async () => {
    const todo = items.filter((i) => !i.result);
    if (!todo.length) return;
    setBusy(true);
    setError(null);
    setProgress(0);
    abort.current = new AbortController();
    try {
      for (let start = 0; start < todo.length; start += CHUNK) {
        const chunk = todo.slice(start, start + CHUNK);
        // Prefix with the item id so results map back unambiguously even if file names repeat.
        const files = chunk.map((i) => new File([i.file], `${i.id}__${i.file.name}`, { type: i.file.type }));
        const res = await api.predictBatch(files, gradcam, abort.current.signal);
        const byId = new Map<string, Partial<Item>>();
        res.results.forEach((r) => byId.set(r.filename.split("__")[0], { result: r }));
        res.errors.forEach((e) => byId.set(e.filename.split("__")[0], { error: e.error }));
        setItems((prev) => prev.map((i) => (byId.has(i.id) ? { ...i, ...byId.get(i.id) } : i)));
        setProgress(Math.min(todo.length, start + CHUNK) / todo.length);
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const done = items.filter((i) => i.result);
  const summary = useMemo(() => {
    const par = done.filter((i) => i.result!.label === "Parasitized").length;
    const reliable = done.filter((i) => i.result!.warnings.length === 0);
    const reliablePar = reliable.filter((i) => i.result!.label === "Parasitized").length;
    return {
      analyzed: done.length,
      parasitized: par,
      uninfected: done.length - par,
      review: done.filter((i) => i.result!.needs_review).length,
      warnings: done.filter((i) => i.result!.warnings.length).length,
      failed: items.filter((i) => i.error).length,
      flagged: done.length - reliable.length,
      parasitemia: reliable.length ? (100 * reliablePar) / reliable.length : 0,
    };
  }, [items]); // eslint-disable-line react-hooks/exhaustive-deps

  const visible = useMemo(() => {
    let v = items.filter((i) => {
      if (filter === "all") return true;
      if (!i.result) return false;
      if (filter === "review") return i.result.needs_review;
      if (filter === "warnings") return i.result.warnings.length > 0;
      return i.result.label === filter;
    });
    if (sortByProb) v = [...v].sort((a, b) => (b.result?.probability_parasitized ?? -1) - (a.result?.probability_parasitized ?? -1));
    return v;
  }, [items, filter, sortByProb]);

  const exportCsv = () => {
    const rows = [["filename", "label", "prob_parasitized", "confidence", "threshold", "needs_review", "warnings"]];
    done.forEach(({ file, result: r }) =>
      rows.push([file.name, r!.label, r!.probability_parasitized.toFixed(4), r!.confidence.toFixed(4), r!.threshold.toFixed(4), String(r!.needs_review), r!.warnings.join(" | ")]),
    );
    const csv = rows.map((r) => r.map((c) => `"${c.replace(/"/g, '""')}"`).join(",")).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "malariascope_results.csv";
    a.click();
  };

  const pending = items.filter((i) => !i.result && !i.error).length;

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-semibold tracking-tight">Screen blood-smear cells</h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-2">
          Upload segmented red-blood-cell images. Each cell gets a calibrated probability of being parasitized, a Grad-CAM
          heatmap of the evidence, and a review flag when the model is unsure. The batch gives an estimated parasitemia.
        </p>
      </section>

      <Dropzone onFiles={addFiles} disabled={busy} />

      {items.length > 0 && (
        <section className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={run}
            disabled={busy || !modelReady || pending === 0}
            className="inline-flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
            {busy ? `Analyzing… ${Math.round(progress * 100)}%` : `Analyze ${pending} image${pending === 1 ? "" : "s"}`}
          </button>
          <label className="inline-flex items-center gap-2 text-sm text-ink-2">
            <input type="checkbox" checked={gradcam} onChange={(e) => setGradcam(e.target.checked)} className="accent-[var(--color-accent)]" />
            Compute Grad-CAM
          </label>
          <span className="text-sm text-ink-3">{items.length} loaded</span>
          <div className="ml-auto flex gap-2">
            <button type="button" onClick={exportCsv} disabled={!done.length} className="btn-secondary inline-flex items-center gap-1.5 rounded-lg border border-line bg-panel px-3 py-2 text-sm hover:bg-black/5 disabled:opacity-40">
              <Download size={15} /> Export CSV
            </button>
            <button type="button" onClick={clear} className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-panel px-3 py-2 text-sm hover:bg-black/5">
              <Trash2 size={15} /> Clear
            </button>
          </div>
        </section>
      )}

      {!modelReady && items.length > 0 && (
        <p className="rounded-lg border border-amber-300 bg-review-bg px-4 py-3 text-sm text-review">
          The API has no model loaded. Train one (see README) and restart the backend.
        </p>
      )}
      {error && <p className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">Request failed: {error}</p>}

      {summary.analyzed > 0 && (
        <section className="grid grid-cols-2 gap-3 md:grid-cols-5">
          <Stat label="Cells analyzed" value={summary.analyzed.toLocaleString()} sub={summary.failed ? `${summary.failed} unreadable` : undefined} />
          <Stat label="Parasitized" value={summary.parasitized.toLocaleString()} swatch="var(--color-parasitized)" />
          <Stat label="Uninfected" value={summary.uninfected.toLocaleString()} swatch="var(--color-uninfected)" />
          <Stat label="Needs review" value={summary.review.toLocaleString()} sub="near the decision threshold" />
          <Stat
            label="Estimated parasitemia"
            value={`${summary.parasitemia.toFixed(1)}%`}
            sub={summary.flagged ? `excludes ${summary.flagged} flagged image${summary.flagged === 1 ? "" : "s"}` : "parasitized ÷ analyzed cells"}
          />
        </section>
      )}

      {items.length > 0 && (
        <section className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            {(
              [
                ["all", `All (${items.length})`],
                ["Parasitized", `Parasitized (${summary.parasitized})`],
                ["Uninfected", `Uninfected (${summary.uninfected})`],
                ["review", `Needs review (${summary.review})`],
                ["warnings", `Input warnings (${summary.warnings})`],
              ] as [Filter, string][]
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setFilter(id)}
                className={`rounded-full border px-3 py-1 text-xs ${filter === id ? "border-ink bg-ink text-white" : "border-line bg-panel text-ink-2 hover:bg-black/5"}`}
              >
                {label}
              </button>
            ))}
            <div className="ml-auto flex items-center gap-4 text-sm text-ink-2">
              <label className="inline-flex items-center gap-2">
                <input type="checkbox" checked={sortByProb} onChange={(e) => setSortByProb(e.target.checked)} /> Sort by P(parasitized)
              </label>
              <button
                type="button"
                onClick={() => {
                  setHeatAll((v) => !v);
                  setHeatOverride({});
                }}
                className="inline-flex items-center gap-1.5 rounded-md border border-line bg-panel px-2.5 py-1 hover:bg-black/5"
              >
                <Flame size={14} /> {heatAll ? "Show originals" : "Show heatmaps"}
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6">
            {visible.map((i) => (
              <ResultCard
                key={i.id}
                name={i.file.name}
                src={i.url}
                result={i.result}
                error={i.error}
                showHeatmap={heatOverride[i.id] ?? heatAll}
                onToggleHeatmap={() => setHeatOverride((o) => ({ ...o, [i.id]: !(o[i.id] ?? heatAll) }))}
              />
            ))}
          </div>
          {visible.length === 0 && <p className="py-8 text-center text-sm text-ink-3">No images match this filter.</p>}
        </section>
      )}
    </div>
  );
}

function Stat({ label, value, sub, swatch }: { label: string; value: string; sub?: string; swatch?: string }) {
  return (
    <div className="rounded-xl border border-line bg-panel p-4">
      <p className="flex items-center gap-1.5 text-xs text-ink-3">
        {swatch && <span className="size-2.5 rounded-full" style={{ background: swatch }} />}
        {label}
      </p>
      <p className="tabular mt-1 text-2xl font-semibold tracking-tight">{value}</p>
      {sub && <p className="mt-0.5 text-[11px] text-ink-3">{sub}</p>}
    </div>
  );
}
