"use client";

import { useEffect, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  LineSeries,
  createChart,
  type CandlestickData,
  type IChartApi,
  type ISeriesApi,
  type LineData,
  type UTCTimestamp,
} from "lightweight-charts";
import type { Candle, Detection } from "@/lib/api";

export interface ChartOverlay {
  key: string;
  label: string;
  price: number;
  color: string;
  dashed?: boolean;
}

export interface CandleChartProps {
  candles: Candle[];
  overlays?: ChartOverlay[];
  height?: number;
}

export function CandleChart({ candles, overlays = [], height = 420 }: CandleChartProps) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const lineRefs = useRef<Map<string, ISeriesApi<"Line">>>(new Map());

  useEffect(() => {
    if (!ref.current) return;
    const isDark = document.documentElement.classList.contains("dark");
    const chart = createChart(ref.current, {
      height,
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: isDark ? "#8b93a7" : "#667085",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: isDark ? "#1d2130" : "#eceef2" },
        horzLines: { color: isDark ? "#1d2130" : "#eceef2" },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, timeVisible: true, secondsVisible: false },
    });
    chartRef.current = chart;
    seriesRef.current = chart.addSeries(CandlestickSeries, {
      upColor: "#26a69a",
      downColor: "#ef5350",
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
      borderVisible: false,
    });
    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      lineRefs.current.clear();
    };
  }, [height]);

  useEffect(() => {
    if (!seriesRef.current || candles.length === 0) return;
    const data: CandlestickData[] = candles.map((c) => ({
      time: (new Date(c.ts).getTime() / 1000) as UTCTimestamp,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    seriesRef.current.setData(data);
    chartRef.current?.timeScale().fitContent();
  }, [candles]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || candles.length === 0) return;
    // sync overlay lines with current toggle set
    const wanted = new Map(overlays.map((o) => [o.key, o]));
    for (const [key, series] of lineRefs.current) {
      if (!wanted.has(key)) {
        chart.removeSeries(series);
        lineRefs.current.delete(key);
      }
    }
    for (const [key, o] of wanted) {
      let series = lineRefs.current.get(key);
      if (!series) {
        series = chart.addSeries(LineSeries, {
          color: o.color,
          lineWidth: 1,
          lineStyle: o.dashed ? 2 : 0,
          title: o.label,
          priceLineVisible: false,
          lastValueVisible: true,
          crosshairMarkerVisible: false,
        });
        lineRefs.current.set(key, series);
      }
      const line: LineData[] = [
        {
          time: (new Date(candles[0].ts).getTime() / 1000) as UTCTimestamp,
          value: o.price,
        },
        {
          time: (new Date(candles[candles.length - 1].ts).getTime() / 1000) as UTCTimestamp,
          value: o.price,
        },
      ];
      series.setData(line);
    }
  }, [overlays, candles]);

  return <div ref={ref} className="w-full" style={{ height }} />;
}

/** Build chart overlays from a detection's levels, honouring toggles. */
export function overlaysForDetection(
  det: Detection | null,
  toggles: Record<string, boolean>
): ChartOverlay[] {
  if (!det) return [];
  const out: ChartOverlay[] = [];
  const levels = det.levels ?? {};
  const defs: { key: string; label: string; color: string; dashed?: boolean; keys: string[] }[] = [
    { key: "breakout", label: "Breakout", color: "#f59e0b", keys: ["breakout", "neckline", "rim"] },
    { key: "support", label: "Support", color: "#26a69a", keys: ["support", "lower", "basin"] },
    { key: "resistance", label: "Resistance", color: "#ef5350", keys: ["resistance", "upper", "head"] },
    { key: "target", label: "Target", color: "#a78bfa", dashed: true, keys: ["target"] },
    { key: "invalidation", label: "Invalidation", color: "#94a3b8", dashed: true, keys: ["invalidation"] },
  ];
  for (const d of defs) {
    if (!toggles[d.key]) continue;
    const found = d.keys.find((k) => typeof levels[k] === "number");
    if (found !== undefined) {
      out.push({ key: d.key, label: d.label, price: levels[found], color: d.color, dashed: d.dashed });
    }
  }
  return out;
}
