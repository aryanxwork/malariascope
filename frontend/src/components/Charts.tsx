import type { ReactNode } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// Chart conventions: 2px lines, hairline recessive grid, legend only when >= 2 series, hover tooltips.
export const SERIES = { primary: "#2a78d6", secondary: "#eb6834", reference: "#a9a7a1" };
const GRID = "#ecebe8";
const AXIS = { stroke: "#d9d8d4", tick: { fill: "#7c7b76", fontSize: 11 } };
const fmt = (v: unknown) => (typeof v === "number" ? v.toFixed(3) : String(v));

export function ChartCard({ title, subtitle, children }: { title: string; subtitle?: ReactNode; children: ReactNode }) {
  return (
    <figure className="rounded-xl border border-line bg-panel p-4">
      <figcaption className="mb-3">
        <p className="text-sm font-semibold">{title}</p>
        {subtitle && <p className="mt-0.5 text-xs text-ink-3">{subtitle}</p>}
      </figcaption>
      <div className="h-64">{children}</div>
    </figure>
  );
}

interface XY {
  x: number;
  y: number;
}

export function CurveChart({
  series,
  xLabel,
  yLabel,
  diagonal,
  point,
  xDomain = [0, 1],
  yDomain = [0, 1],
  verticalAt,
  integerX,
}: {
  series: { name: string; color: string; data: XY[]; dots?: boolean }[];
  xLabel: string;
  yLabel: string;
  diagonal?: boolean;
  point?: XY & { label: string };
  xDomain?: [number | "auto", number | "auto"];
  yDomain?: [number | "auto", number | "auto"];
  verticalAt?: { x: number; label: string };
  integerX?: boolean;
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
        <CartesianGrid stroke={GRID} strokeWidth={1} />
        <XAxis
          type="number"
          dataKey="x"
          domain={xDomain}
          allowDecimals={!integerX}
          {...AXIS}
          label={{ value: xLabel, position: "insideBottom", offset: -10, fill: "#7c7b76", fontSize: 11 }}
        />
        <YAxis
          type="number"
          domain={yDomain}
          width={44}
          {...AXIS}
          tickFormatter={(v: number) => String(Number(v.toFixed(3)))}
          label={{ value: yLabel, angle: -90, position: "insideLeft", offset: 12, fill: "#7c7b76", fontSize: 11, style: { textAnchor: "middle" } }}
        />
        <Tooltip
          formatter={(v, n) => [fmt(v), n]}
          labelFormatter={(l) => `${xLabel}: ${fmt(l)}`}
          contentStyle={{ borderRadius: 8, borderColor: "#e6e5e1", fontSize: 12 }}
        />
        {diagonal && (
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={SERIES.reference} strokeWidth={1} ifOverflow="hidden" />
        )}
        {verticalAt && (
          <ReferenceLine x={verticalAt.x} stroke={SERIES.reference} strokeWidth={1} label={{ value: verticalAt.label, position: "insideTopRight", fill: "#7c7b76", fontSize: 10 }} />
        )}
        {series.map((s) => (
          <Line
            key={s.name}
            name={s.name}
            data={s.data}
            dataKey="y"
            type="linear"
            stroke={s.color}
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
            dot={s.dots ? { r: 4, fill: s.color, stroke: "#fff", strokeWidth: 2 } : false}
            activeDot={{ r: 5, stroke: "#fff", strokeWidth: 2 }}
            isAnimationActive={false}
          />
        ))}
        {point && <ReferenceDot x={point.x} y={point.y} r={6} fill={SERIES.secondary} stroke="#fff" strokeWidth={2} label={{ value: point.label, position: "right", fill: "#52514e", fontSize: 11 }} />}
        {series.length >= 2 && <Legend verticalAlign="top" height={24} iconType="plainline" wrapperStyle={{ fontSize: 12, color: "#52514e" }} />}
      </LineChart>
    </ResponsiveContainer>
  );
}
