import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

interface PlotlyChartProps {
  data: Record<string, unknown>[];
  layout: Record<string, unknown>;
  className?: string;
}

export function PlotlyChart({ data, layout, className = 'w-full h-[380px]' }: PlotlyChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    Plotly.react(
      el,
      data,
      {
        paper_bgcolor: '#FFFFFF',
        plot_bgcolor: '#FFFFFF',
        font: { family: 'Public Sans, system-ui, sans-serif', color: '#0F172A', size: 12 },
        ...layout,
      },
      {
        responsive: true,
        displaylogo: false,
      }
    );

    const handleResize = () => {
      if (el) Plotly.Plots.resize(el);
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [data, layout]);

  useEffect(() => {
    const el = containerRef.current;
    return () => {
      if (el) Plotly.purge(el);
    };
  }, []);

  return <div ref={containerRef} className={className} />;
}
