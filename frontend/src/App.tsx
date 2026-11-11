import { lazy, Suspense, useEffect, useState } from "react";
import { Microscope, Activity, BarChart3, Info } from "lucide-react";
import { api, type Health } from "./lib/api";
import AnalyzePage from "./pages/Analyze";
import AboutPage from "./pages/About";

// Charts (recharts) are only needed on the performance page -> separate chunk.
const PerformancePage = lazy(() => import("./pages/Performance"));

type Tab = "analyze" | "performance" | "about";
const TABS: { id: Tab; label: string; icon: typeof Activity }[] = [
  { id: "analyze", label: "Analyze", icon: Activity },
  { id: "performance", label: "Model performance", icon: BarChart3 },
  { id: "about", label: "How it works", icon: Info },
];

const tabFromHash = (): Tab => {
  const h = window.location.hash.replace("#/", "") as Tab;
  return TABS.some((t) => t.id === h) ? h : "analyze";
};

export default function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    const load = () =>
      api
        .health()
        .then((h) => {
          setHealth(h);
          setHealthError(null);
        })
        .catch((e: Error) => setHealthError(e.message));
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-line bg-panel/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 py-3 sm:gap-6 sm:px-6">
          <a href="#/analyze" className="flex items-center gap-2.5">
            <span className="grid size-9 place-items-center rounded-lg bg-ink text-white">
              <Microscope size={18} />
            </span>
            <span className="leading-tight">
              <span className="block text-[15px] font-semibold tracking-tight">MalariaScope</span>
              <span className="hidden text-xs text-ink-3 sm:block">Explainable thin-smear screening</span>
            </span>
          </a>
          <nav className="flex gap-1" aria-label="Main">
            {TABS.map(({ id, label, icon: Icon }) => (
              <a
                key={id}
                href={`#/${id}`}
                aria-current={tab === id ? "page" : undefined}
                className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition ${
                  tab === id ? "bg-ink text-white" : "text-ink-2 hover:bg-black/5"
                }`}
              >
                <Icon size={15} />
                <span className="hidden sm:inline">{label}</span>
              </a>
            ))}
          </nav>
          <div className="ml-auto">
            <StatusPill health={health} error={healthError} />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        {tab === "analyze" && <AnalyzePage modelReady={!!health?.model_loaded} />}
        {tab === "performance" && (
          <Suspense fallback={<p className="text-sm text-ink-3">Loading…</p>}>
            <PerformancePage />
          </Suspense>
        )}
        {tab === "about" && <AboutPage />}
      </main>

      <footer className="mx-auto max-w-7xl px-6 pb-10 text-xs text-ink-3">
        Research and education prototype — not a medical device and not for clinical diagnosis. Data: NIH/NLM
        malaria cell images (Rajaraman et al., PeerJ 2018).
      </footer>
    </div>
  );
}

function StatusPill({ health, error }: { health: Health | null; error: string | null }) {
  const ok = health?.model_loaded;
  const text = error ? "API offline" : !health ? "Connecting…" : ok ? `Model: ${health.model}` : "No model loaded";
  return (
    <span
      title={error ?? health?.detail ?? health?.run ?? ""}
      className="inline-flex items-center gap-2 rounded-full border border-line bg-panel px-3 py-1 text-xs text-ink-2"
    >
      <span className={`size-2 rounded-full ${ok ? "bg-emerald-500" : error ? "bg-red-500" : "bg-amber-500"}`} />
      <span className="hidden md:inline">{text}</span>
    </span>
  );
}
