import { useMemo, useState } from 'react';
import { Play, Sparkles, Download, CheckCircle2, AlertTriangle, RefreshCw } from 'lucide-react';
import {
  CLUSTER_COLORS,
  CentroidRow,
  ClusterMetrics,
  ClusterProfile,
  ClusteringAlgorithm,
  DataRow,
  DistanceMetric,
  FivePillarsStrategy,
  KMetricPoint,
  LinkageMethod,
  PCAProjection,
  ScalingMethod,
} from '../types/clustering';
import { validateHierarchicalCombination } from '../utils/mlEngine';
import { PlotlyChart } from './PlotlyChart';

/* -------------------------------------------------------------------------- */
/* PASO 7: Determinación del número óptimo de clusters (K)                     */
/* -------------------------------------------------------------------------- */
interface Step7Props {
  scalingMethod: ScalingMethod;
  kPoints: KMetricPoint[];
  bestSilhouetteK: number;
  elbowK: number;
  selectedK: number;
  businessNote: string;
  onScalingChange: (m: ScalingMethod) => void;
  onRecalculateK: () => void;
  onSelectK: (k: number) => void;
  onBusinessNoteChange: (note: string) => void;
}

export function Step7OptimalK({
  scalingMethod,
  kPoints,
  bestSilhouetteK,
  elbowK,
  selectedK,
  businessNote,
  onScalingChange,
  onRecalculateK,
  onSelectK,
  onBusinessNoteChange,
}: Step7Props) {
  const elbowData = useMemo(
    () => [
      {
        x: kPoints.map((p) => p.k),
        y: kPoints.map((p) => p.inertia),
        mode: 'lines+markers',
        type: 'scatter',
        name: 'Inercia',
        line: { color: '#0033A0', width: 2.5 },
        marker: { size: 8, color: '#0033A0' },
      },
    ],
    [kPoints]
  );

  const silData = useMemo(
    () => [
      {
        x: kPoints.map((p) => p.k),
        y: kPoints.map((p) => p.silhouette),
        mode: 'lines+markers',
        type: 'scatter',
        name: 'Silhouette Score',
        line: { color: '#00A99D', width: 2.5 },
        marker: { size: 8, color: '#00A99D' },
      },
    ],
    [kPoints]
  );

  const maxK = kPoints.length ? kPoints[kPoints.length - 1].k : 10;

  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div className="flex-1 max-w-sm">
          <label className="block text-xs font-semibold text-slate-700 mb-1.5">
            Método de escalamiento automático
          </label>
          <select
            value={scalingMethod}
            onChange={(e) => onScalingChange(e.target.value as ScalingMethod)}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-[#0033A0]"
          >
            <option value="StandardScaler">StandardScaler (Media 0, Desv. Est. 1)</option>
            <option value="MinMaxScaler">MinMaxScaler (Rango 0 a 1)</option>
          </select>
        </div>
        <button
          type="button"
          onClick={onRecalculateK}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 px-4 py-2.5 text-sm font-medium text-slate-800 transition-colors"
        >
          <RefreshCw className="w-4 h-4 text-[#0033A0]" />
          <span>Recalcular métricas (K=2..10)</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <PlotlyChart
            data={elbowData}
            layout={{
              title: { text: `A. Método del Codo (Inflexión sugerida: K = ${elbowK})`, font: { size: 14 } },
              xaxis: { title: 'Número de clusters (K)', dtick: 1 },
              yaxis: { title: 'Inercia intra-cluster' },
              shapes: [
                {
                  type: 'line',
                  x0: elbowK,
                  x1: elbowK,
                  y0: 0,
                  y1: 1,
                  yref: 'paper',
                  line: { color: '#78BE20', width: 2, dash: 'dash' },
                },
              ],
              margin: { l: 60, r: 20, t: 50, b: 50 },
            }}
            className="w-full h-[330px]"
          />
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <PlotlyChart
            data={silData}
            layout={{
              title: { text: `B. Coeficiente de Silueta (Óptimo técnico: K = ${bestSilhouetteK})`, font: { size: 14 } },
              xaxis: { title: 'Número de clusters (K)', dtick: 1 },
              yaxis: { title: 'Silhouette Score' },
              shapes: [
                {
                  type: 'line',
                  x0: bestSilhouetteK,
                  x1: bestSilhouetteK,
                  y0: 0,
                  y1: 1,
                  yref: 'paper',
                  line: { color: '#EAB308', width: 2, dash: 'dash' },
                },
              ],
              margin: { l: 60, r: 20, t: 50, b: 50 },
            }}
            className="w-full h-[330px]"
          />
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
        <div>
          <h3 className="text-base font-semibold text-slate-900">C. Criterio de negocio y decisión final</h3>
          <p className="text-sm text-slate-600 mt-1">
            Recomendación técnica: Silueta máxima en <strong>K = {bestSilhouetteK}</strong> e inflexión de codo en{' '}
            <strong>K = {elbowK}</strong>. Ajuste el número definitivo considerando su capacidad operativa, recursos comerciales y facilidad de implementación.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          <div>
            <div className="flex items-center justify-between text-sm mb-2">
              <span className="font-medium text-slate-700">Clusters seleccionados (K)</span>
              <span className="font-mono font-bold text-lg text-[#0033A0] tabular-nums">{selectedK}</span>
            </div>
            <input
              type="range"
              min={2}
              max={Math.max(2, maxK)}
              value={Math.min(selectedK, Math.max(2, maxK))}
              onChange={(e) => onSelectK(Number(e.target.value))}
              className="w-full accent-[#0033A0] cursor-pointer"
            />
            <div className="flex justify-between text-xs text-slate-500 font-mono mt-1">
              <span>2</span>
              <span>{Math.max(2, maxK)}</span>
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1.5">
              Justificación operativa / comercial (opcional)
            </label>
            <input
              type="text"
              value={businessNote}
              onChange={(e) => onBusinessNoteChange(e.target.value)}
              placeholder="Ej. Capacidad comercial para operar 4 estrategias simultáneas."
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-[#0033A0]"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* PASO 8: Selección del algoritmo y métrica de distancia                      */
/* -------------------------------------------------------------------------- */
interface Step8Props {
  algorithm: ClusteringAlgorithm;
  metric: DistanceMetric;
  linkage: LinkageMethod;
  nInit: number;
  randomState: number;
  maxIter: number;
  tol: number;
  onAlgorithmChange: (a: ClusteringAlgorithm) => void;
  onMetricChange: (m: DistanceMetric) => void;
  onLinkageChange: (l: LinkageMethod) => void;
  onKMeansParamsChange: (field: 'nInit' | 'randomState' | 'maxIter' | 'tol', val: number) => void;
}

export function Step8AlgorithmSelection({
  algorithm,
  metric,
  linkage,
  nInit,
  randomState,
  maxIter,
  tol,
  onAlgorithmChange,
  onMetricChange,
  onLinkageChange,
  onKMeansParamsChange,
}: Step8Props) {
  const hierarchicalCheck = validateHierarchicalCombination(metric, linkage);

  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-5">
        <div>
          <h2 className="text-base font-semibold text-slate-900">Algoritmo de segmentación</h2>
          <p className="text-sm text-slate-600 mt-1">
            Seleccione el enfoque matemático y configure los hiperparámetros de convergencia o enlace.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {(['K-Means', 'Jerárquico / Aglomerativo'] as ClusteringAlgorithm[]).map((opt) => (
            <button
              key={opt}
              type="button"
              onClick={() => onAlgorithmChange(opt)}
              className={`text-left rounded-xl border p-4 transition-colors ${
                algorithm === opt
                  ? 'border-[#0033A0] bg-[#0033A0]/5 text-slate-900'
                  : 'border-slate-200 bg-white hover:bg-slate-50 text-slate-700'
              }`}
            >
              <div className="font-semibold text-sm">{opt}</div>
              <div className="text-xs text-slate-500 mt-1">
                {opt === 'K-Means'
                  ? 'Particional mediante distancia euclidiana e inicialización K-Means++.'
                  : 'Aglomerativo con selección de métrica (Euclidiana, Manhattan, Coseno) y enlace.'}
              </div>
            </button>
          ))}
        </div>

        {algorithm === 'K-Means' ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-2">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Inicializaciones (n_init)
              </label>
              <input
                type="number"
                min={1}
                max={50}
                value={nInit}
                onChange={(e) => onKMeansParamsChange('nInit', Math.max(1, Number(e.target.value)))}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-mono tabular-nums"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Semilla aleatoria
              </label>
              <input
                type="number"
                min={0}
                value={randomState}
                onChange={(e) => onKMeansParamsChange('randomState', Math.max(0, Number(e.target.value)))}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-mono tabular-nums"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Máx. iteraciones
              </label>
              <input
                type="number"
                min={20}
                max={1000}
                value={maxIter}
                onChange={(e) => onKMeansParamsChange('maxIter', Math.max(20, Number(e.target.value)))}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-mono tabular-nums"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Tolerancia convergencia
              </label>
              <input
                type="number"
                step="0.0001"
                min={0.000001}
                value={tol}
                onChange={(e) => onKMeansParamsChange('tol', Math.max(1e-6, Number(e.target.value)))}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-mono tabular-nums"
              />
            </div>
          </div>
        ) : (
          <div className="space-y-4 pt-2">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Métrica de distancia
                </label>
                <select
                  value={metric}
                  onChange={(e) => onMetricChange(e.target.value as DistanceMetric)}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900"
                >
                  <option value="euclidean">Distancia Euclidiana</option>
                  <option value="manhattan">Distancia Manhattan</option>
                  <option value="cosine">Distancia Coseno</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Método de enlace (Linkage)
                </label>
                <select
                  value={linkage}
                  onChange={(e) => onLinkageChange(e.target.value as LinkageMethod)}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900"
                >
                  <option value="ward">Ward (minimiza varianza intra-cluster)</option>
                  <option value="complete">Complete (enlace máximo)</option>
                  <option value="average">Average (enlace promedio)</option>
                  <option value="single">Single (enlace mínimo)</option>
                </select>
              </div>
            </div>

            {hierarchicalCheck.valid ? (
              <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-2.5 text-sm text-emerald-900">
                <CheckCircle2 className="w-4 h-4 text-[#65A30D]" />
                <span>{hierarchicalCheck.message}</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-2.5 text-sm text-red-800">
                <AlertTriangle className="w-4 h-4 text-red-600" />
                <span>{hierarchicalCheck.message}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* PASO 9: Ejecución del clustering                                            */
/* -------------------------------------------------------------------------- */
interface Step9Props {
  k: number;
  algorithm: ClusteringAlgorithm;
  rowCount: number;
  labels: number[] | null;
  metrics: ClusterMetrics | null;
  labeledRows: DataRow[];
  columns: string[];
  onExecute: () => void;
}

export function Step9Execution({
  k,
  algorithm,
  rowCount,
  labels,
  metrics,
  labeledRows,
  columns,
  onExecute,
}: Step9Props) {
  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-slate-900">Ejecución del modelo de segmentación</h2>
          <p className="text-sm text-slate-600 mt-1">
            Algoritmo: <strong>{algorithm}</strong> · Segmentos (K): <strong>{k}</strong> · Registros preparados:{' '}
            <strong className="font-mono tabular-nums">{rowCount.toLocaleString('es-CO')}</strong>
          </p>
        </div>
        <button
          type="button"
          onClick={onExecute}
          className="inline-flex items-center justify-center gap-2 rounded-lg bg-[#0033A0] hover:bg-[#002880] px-5 py-2.5 text-sm font-medium text-white transition-colors whitespace-nowrap shrink-0"
        >
          <Play className="w-4 h-4" />
          <span>{labels ? 'Re-ejecutar clustering' : 'Ejecutar clustering'}</span>
        </button>
      </div>

      {metrics && labels && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-white border border-slate-200 rounded-xl p-5">
              <div className="text-xs font-medium text-slate-500">Segmentos generados</div>
              <div className="text-2xl font-bold text-slate-900 tabular-nums mt-1">{k}</div>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-5">
              <div className="text-xs font-medium text-slate-500">Registros clasificados</div>
              <div className="text-2xl font-bold text-slate-900 tabular-nums mt-1">
                {labels.length.toLocaleString('es-CO')}
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-5">
              <div className="text-xs font-medium text-slate-500">Silhouette Score</div>
              <div className="text-2xl font-bold text-[#00A99D] tabular-nums mt-1">
                {metrics.silhouette.toFixed(4)}
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-5">
              <div className="text-xs font-medium text-slate-500">Davies-Bouldin</div>
              <div className="text-2xl font-bold text-slate-900 tabular-nums mt-1">
                {metrics.daviesBouldin.toFixed(4)}
              </div>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-3">
            <h3 className="text-base font-semibold text-slate-900">
              Registros clasificados con etiqueta <code className="font-mono text-[#0033A0]">Cluster</code> (muestra de 15 filas)
            </h3>
            <div className="overflow-x-auto border border-slate-200 rounded-lg">
              <table className="min-w-full text-xs text-left">
                <thead className="bg-[#0033A0] text-white">
                  <tr>
                    <th className="px-3.5 py-2.5 font-semibold">Cluster</th>
                    {columns.map((c) => (
                      <th key={c} className="px-3.5 py-2.5 font-semibold whitespace-nowrap">
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200">
                  {labeledRows.slice(0, 15).map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-50">
                      <td className="px-3.5 py-2 font-mono font-semibold text-[#0033A0]">
                        Cluster {String(row.Cluster)}
                      </td>
                      {columns.map((c) => (
                        <td key={c} className="px-3.5 py-2 font-mono tabular-nums whitespace-nowrap">
                          {row[c] === null ? '' : String(row[c])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* PASO 10: Visualización de centroides y perfiles                             */
/* -------------------------------------------------------------------------- */
interface Step10Props {
  centroids: CentroidRow[];
  numericCols: string[];
  labels: number[];
  pca: PCAProjection;
  profiles: Record<number, ClusterProfile> | null;
}

export function Step10CentroidsVisuals({
  centroids,
  numericCols,
  labels,
  pca,
  profiles,
}: Step10Props) {
  const [activeTab, setActiveTab] = useState<'pca' | 'radar'>('pca');

  const pcaTraces = useMemo(() => {
    const traces: Record<string, unknown>[] = centroids.map((c) => {
      const xVals: number[] = [];
      const yVals: number[] = [];
      for (let i = 0; i < labels.length; i++) {
        if (labels[i] === c.clusterId) {
          xVals.push(pca.points[i][0]);
          yVals.push(pca.points[i][1]);
        }
      }
      const labelName = profiles?.[c.clusterId]?.nombreComercial || `Cluster ${c.clusterId}`;
      return {
        x: xVals,
        y: yVals,
        mode: 'markers',
        type: 'scatter',
        name: `${labelName} (n=${c.n})`,
        marker: {
          size: 8,
          color: CLUSTER_COLORS[c.clusterId % CLUSTER_COLORS.length],
          opacity: 0.72,
          line: { width: 0.6, color: '#FFFFFF' },
        },
      };
    });

    traces.push({
      x: pca.centroidPoints.map((p) => p[0]),
      y: pca.centroidPoints.map((p) => p[1]),
      mode: 'markers+text',
      type: 'scatter',
      name: 'Centroides',
      text: centroids.map((c) => `C${c.clusterId}`),
      textposition: 'top center',
      marker: {
        size: 16,
        symbol: 'diamond',
        color: '#FCD116',
        line: { width: 2, color: '#0F172A' },
      },
    });
    return traces;
  }, [centroids, labels, pca, profiles]);

  const radarTraces = useMemo(() => {
    return centroids.map((c) => {
      const rVals = numericCols.map((col) => {
        const all = centroids.map((item) => item.values[col]);
        const minV = Math.min(...all);
        const maxV = Math.max(...all);
        return maxV - minV > 0 ? (c.values[col] - minV) / (maxV - minV) : 0.5;
      });
      rVals.push(rVals[0]);
      const theta = [...numericCols, numericCols[0]];
      const custom = [
        ...numericCols.map((col) => `${col}: ${c.values[col].toLocaleString('es-CO')}`),
        `${numericCols[0]}: ${c.values[numericCols[0]].toLocaleString('es-CO')}`,
      ];
      const name = profiles?.[c.clusterId]?.nombreComercial || `Cluster ${c.clusterId}`;
      return {
        type: 'scatterpolar',
        r: rVals,
        theta,
        fill: 'toself',
        opacity: 0.62,
        name,
        customdata: custom,
        hovertemplate: `<b>${name}</b><br>%{customdata}<br>Índice normalizado: %{r:.2f}<extra></extra>`,
        line: { color: CLUSTER_COLORS[c.clusterId % CLUSTER_COLORS.length], width: 2 },
      };
    });
  }, [centroids, numericCols, profiles]);

  const var1 = (pca.explainedVarianceRatio[0] * 100).toFixed(1);
  const var2 = (pca.explainedVarianceRatio[1] * 100).toFixed(1);

  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
        <div>
          <h2 className="text-base font-semibold text-slate-900">Tabla de centroides desescalados (unidades originales)</h2>
          <p className="text-sm text-slate-600 mt-1">
            Calculados como las medias aritméticas reales de cada grupo sobre las variables originales de negocio.
          </p>
        </div>

        <div className="overflow-x-auto border border-slate-200 rounded-lg">
          <table className="min-w-full text-sm text-right">
            <thead className="bg-[#0033A0] text-white text-xs">
              <tr>
                <th className="px-4 py-3 text-left font-semibold">Cluster</th>
                {numericCols.map((col) => (
                  <th key={col} className="px-4 py-3 font-semibold whitespace-nowrap">
                    {col}
                  </th>
                ))}
                <th className="px-4 py-3 font-semibold">N</th>
                <th className="px-4 py-3 font-semibold">Porcentaje</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 font-mono tabular-nums">
              {centroids.map((c) => (
                <tr key={c.clusterId} className="hover:bg-slate-50">
                  <td className="px-4 py-2.5 text-left font-sans font-semibold text-slate-900">
                    Cluster {c.clusterId}
                  </td>
                  {numericCols.map((col) => (
                    <td key={col} className="px-4 py-2.5">
                      {c.values[col].toLocaleString('es-CO', {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                      })}
                    </td>
                  ))}
                  <td className="px-4 py-2.5 font-semibold">{c.n.toLocaleString('es-CO')}</td>
                  <td className="px-4 py-2.5 font-semibold text-[#0033A0]">{c.percentage.toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-200 pb-3">
          <button
            type="button"
            onClick={() => setActiveTab('pca')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === 'pca'
                ? 'bg-[#0033A0] text-white'
                : 'text-slate-600 hover:bg-slate-100'
            }`}
          >
            Gráfico PCA 2D
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('radar')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === 'radar'
                ? 'bg-[#0033A0] text-white'
                : 'text-slate-600 hover:bg-slate-100'
            }`}
          >
            Gráfico de Radar
          </button>
        </div>

        {activeTab === 'pca' ? (
          <PlotlyChart
            data={pcaTraces}
            layout={{
              title: {
                text: `Proyección PCA 2D (Varianza explicada: ${(Number(var1) + Number(var2)).toFixed(1)}%)`,
                font: { size: 15 },
              },
              xaxis: { title: `Componente Principal 1 (${var1}%)` },
              yaxis: { title: `Componente Principal 2 (${var2}%)` },
              legend: { orientation: 'h', y: -0.2 },
              margin: { l: 55, r: 25, t: 55, b: 85 },
            }}
            className="w-full h-[460px]"
          />
        ) : (
          <PlotlyChart
            data={radarTraces}
            layout={{
              title: {
                text: 'Comparación Multivariada de Perfiles (Normalizado 0–1)',
                font: { size: 15 },
              },
              polar: { radialaxis: { visible: true, range: [0, 1.05] } },
              legend: { orientation: 'h', y: -0.2 },
              margin: { l: 60, r: 60, t: 60, b: 85 },
            }}
            className="w-full h-[460px]"
          />
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* PASO 11: Perfilamiento automático y nombres comerciales                     */
/* -------------------------------------------------------------------------- */
interface Step11Props {
  sector: string;
  hasApiKey: boolean;
  aiModeUsed: string;
  loading: boolean;
  errorMsg: string | null;
  centroids: CentroidRow[];
  profiles: Record<number, ClusterProfile> | null;
  onGenerateProfiles: () => void;
}

export function Step11Profiling({
  sector,
  hasApiKey,
  aiModeUsed,
  loading,
  errorMsg,
  centroids,
  profiles,
  onGenerateProfiles,
}: Step11Props) {
  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-slate-900">Perfilamiento comercial de segmentos</h2>
          <p className="text-sm text-slate-600 mt-1">
            Sector: <strong>{sector}</strong> · Modalidad disponible:{' '}
            <strong>{hasApiKey ? 'Inteligencia Artificial (OpenAI API)' : 'Simulación / Plantilla Estructurada'}</strong>
          </p>
        </div>
        <button
          type="button"
          disabled={loading}
          onClick={onGenerateProfiles}
          className="inline-flex items-center justify-center gap-2 rounded-lg bg-[#0033A0] hover:bg-[#002880] disabled:opacity-60 px-5 py-2.5 text-sm font-medium text-white transition-colors whitespace-nowrap shrink-0"
        >
          <Sparkles className="w-4 h-4" />
          <span>{loading ? 'Generando perfiles...' : profiles ? 'Regenerar perfiles' : 'Generar perfiles comerciales'}</span>
        </button>
      </div>

      {errorMsg && (
        <div className="flex items-start gap-2.5 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <AlertTriangle className="w-4 h-4 text-amber-600 mt-0.5 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {profiles && (
        <div className="space-y-4">
          <div className="text-xs text-slate-500">Modalidad aplicada: {aiModeUsed}</div>
          {centroids.map((c) => {
            const p = profiles[c.clusterId];
            if (!p) return null;
            return (
              <article
                key={c.clusterId}
                className="bg-white border border-slate-200 rounded-xl p-6 space-y-3"
                style={{ borderTop: `3px solid ${CLUSTER_COLORS[c.clusterId % CLUSTER_COLORS.length]}` }}
              >
                <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-slate-100 pb-3">
                  <h3 className="text-base font-semibold text-slate-900">
                    Cluster {c.clusterId}: {p.nombreComercial}
                  </h3>
                  <span className="text-xs font-mono tabular-nums text-[#0033A0] font-semibold">
                    {c.n.toLocaleString('es-CO')} registros · {c.percentage.toFixed(1)}%
                  </span>
                </div>
                <p className="text-sm text-slate-700">
                  <strong className="text-slate-900">Descripción del perfil:</strong> {p.descripcionPerfil}
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-slate-700 pt-1">
                  <div>
                    <strong className="text-slate-900 block mb-1">Comportamiento predominante</strong>
                    <p>{p.comportamientoPredominante}</p>
                  </div>
                  <div>
                    <strong className="text-slate-900 block mb-1">Necesidades relevantes</strong>
                    <p>{p.necesidadesRelevantes}</p>
                  </div>
                </div>
                <p className="text-sm text-slate-700">
                  <strong className="text-slate-900">Oportunidades comerciales:</strong> {p.oportunidadesComerciales}
                </p>
                <div className="text-sm text-slate-700">
                  <strong className="text-slate-900 block mb-1">Características diferenciadoras</strong>
                  <ul className="list-disc pl-5 space-y-1">
                    {p.caracteristicasDiferenciadoras.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* PASO 12: Generación de estrategias comerciales (5 pilares)                  */
/* -------------------------------------------------------------------------- */
interface Step12Props {
  centroids: CentroidRow[];
  profiles: Record<number, ClusterProfile>;
  strategies: Record<number, FivePillarsStrategy>;
}

export function Step12Strategies({ centroids, profiles, strategies }: Step12Props) {
  const [activeCluster, setActiveCluster] = useState<number>(centroids[0]?.clusterId ?? 0);
  const currentCentroid = centroids.find((c) => c.clusterId === activeCluster) || centroids[0];
  const currentProfile = profiles[currentCentroid.clusterId];
  const currentStrategy = strategies[currentCentroid.clusterId];

  const pillars: (keyof FivePillarsStrategy)[] = [
    'Producto',
    'Precio',
    'Innovación',
    'Comunicación',
    'Rentabilidad',
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2">
        {centroids.map((c) => {
          const isSelected = c.clusterId === currentCentroid.clusterId;
          return (
            <button
              key={c.clusterId}
              type="button"
              onClick={() => setActiveCluster(c.clusterId)}
              className={`px-3.5 py-2 rounded-lg text-sm font-medium border transition-colors whitespace-nowrap shrink-0 ${
                isSelected
                  ? 'bg-[#0033A0] text-white border-[#0033A0]'
                  : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-50'
              }`}
            >
              Cluster {c.clusterId}: {profiles[c.clusterId]?.nombreComercial}
            </button>
          );
        })}
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-5">
        <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-slate-200 pb-3">
          <h2 className="text-lg font-semibold text-slate-900">{currentProfile?.nombreComercial}</h2>
          <span className="text-sm font-mono tabular-nums text-[#0033A0] font-semibold">
            {currentCentroid.n.toLocaleString('es-CO')} registros · {currentCentroid.percentage.toFixed(1)}%
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {pillars.map((pillar) => (
            <div key={pillar} className="border border-slate-200 rounded-lg p-4 bg-slate-50/50">
              <h3 className="text-sm font-semibold text-[#0033A0] mb-2">{pillar}</h3>
              <ul className="list-disc pl-5 space-y-1.5 text-sm text-slate-700">
                {(currentStrategy?.[pillar] || []).map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* PASO 13: Exportación del informe ejecutivo en HTML                          */
/* -------------------------------------------------------------------------- */
interface Step13Props {
  projectName: string;
  sector: string;
  algorithm: ClusteringAlgorithm;
  k: number;
  modeledCount: number;
  onDownloadHtml: () => void;
}

export function Step13ExportReport({
  projectName,
  sector,
  algorithm,
  k,
  modeledCount,
  onDownloadHtml,
}: Step13Props) {
  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-5">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Exportar Informe Ejecutivo Autocontenido (.html)</h2>
          <p className="text-sm text-slate-600 mt-1">
            El documento generado integra portada institucional, resumen de preparación de datos, metodología,
            métricas de evaluación, tabla de centroides desescalados, gráficos interactivos PCA 2D y Radar, perfiles comerciales,
            estrategias de los cinco pilares y conclusiones ejecutivas.
          </p>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          <div className="border border-slate-200 rounded-lg p-3.5">
            <div className="text-xs text-slate-500">Proyecto</div>
            <div className="font-semibold text-slate-900 mt-0.5 truncate">{projectName}</div>
          </div>
          <div className="border border-slate-200 rounded-lg p-3.5">
            <div className="text-xs text-slate-500">Sector económico</div>
            <div className="font-semibold text-slate-900 mt-0.5 truncate">{sector}</div>
          </div>
          <div className="border border-slate-200 rounded-lg p-3.5">
            <div className="text-xs text-slate-500">Modelo</div>
            <div className="font-semibold text-slate-900 mt-0.5">
              {algorithm} (K={k})
            </div>
          </div>
          <div className="border border-slate-200 rounded-lg p-3.5">
            <div className="text-xs text-slate-500">Registros clasificados</div>
            <div className="font-semibold text-slate-900 font-mono tabular-nums mt-0.5">
              {modeledCount.toLocaleString('es-CO')}
            </div>
          </div>
        </div>

        <button
          type="button"
          onClick={onDownloadHtml}
          className="inline-flex items-center justify-center gap-2.5 rounded-lg bg-[#0033A0] hover:bg-[#002880] px-6 py-3 text-sm font-semibold text-white transition-colors"
        >
          <Download className="w-4 h-4" />
          <span>Descargar Informe Ejecutivo (.html)</span>
        </button>
      </div>
    </div>
  );
}
