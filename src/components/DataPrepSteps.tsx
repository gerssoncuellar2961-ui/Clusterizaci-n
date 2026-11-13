import React, { useMemo, useState } from 'react';
import { Upload, Database, Trash2, CheckCircle2, AlertTriangle, Filter } from 'lucide-react';
import {
  BusinessRuleRange,
  ColumnInspection,
  DataRow,
  DescriptiveStatsRow,
  ECONOMIC_SECTORS,
  EncodingOption,
  ImputationStrategy,
  SeparatorOption,
} from '../types/clustering';
import { PlotlyChart } from './PlotlyChart';

/* -------------------------------------------------------------------------- */
/* PASO 1: Carga de archivos multiformato                                      */
/* -------------------------------------------------------------------------- */
interface Step1Props {
  fileName: string | null;
  rowCount: number;
  colCount: number;
  separator: SeparatorOption;
  encoding: EncodingOption;
  onSeparatorChange: (s: SeparatorOption) => void;
  onEncodingChange: (e: EncodingOption) => void;
  onFileUpload: (file: File) => Promise<void>;
  onLoadSample: () => void;
}

export function Step1Upload({
  fileName,
  rowCount,
  colCount,
  separator,
  encoding,
  onSeparatorChange,
  onEncodingChange,
  onFileUpload,
  onLoadSample,
}: Step1Props) {
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setErrorMsg(null);
    setLoading(true);
    try {
      await onFileUpload(file);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Error al procesar el archivo.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-5">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Importar base de datos empresarial</h2>
          <p className="text-sm text-slate-600 mt-1">
            Formatos compatibles: CSV, TXT, XLS y XLSX. Configure el delimitador y la codificación antes de seleccionar archivos de texto plano.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1.5">
              Separador para CSV / TXT
            </label>
            <select
              value={separator}
              onChange={(e) => onSeparatorChange(e.target.value as SeparatorOption)}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-[#0033A0]"
            >
              <option value=",">Coma (,)</option>
              <option value=";">Punto y coma (;)</option>
              <option value="	">Tabulador (\t)</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1.5">
              Codificación de caracteres
            </label>
            <select
              value={encoding}
              onChange={(e) => onEncodingChange(e.target.value as EncodingOption)}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-[#0033A0]"
            >
              <option value="utf-8">UTF-8 (Estándar internacional)</option>
              <option value="iso-8859-1">Latin-1 / ISO-8859-1 (Acentos Windows)</option>
            </select>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 pt-2">
          <label className="flex-1 cursor-pointer inline-flex items-center justify-center gap-2.5 rounded-lg border border-dashed border-slate-300 bg-slate-50 hover:bg-slate-100 px-5 py-4 text-sm font-medium text-slate-800 transition-colors">
            <Upload className="w-4 h-4 text-[#0033A0]" />
            <span>{loading ? 'Analizando archivo...' : 'Seleccionar archivo CSV, TXT, XLS o XLSX'}</span>
            <input
              type="file"
              accept=".csv,.txt,.xls,.xlsx"
              onChange={handleChange}
              className="hidden"
            />
          </label>

          <button
            type="button"
            onClick={() => {
              setErrorMsg(null);
              onLoadSample();
            }}
            className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 px-4 py-3 text-sm font-medium text-slate-800 transition-colors whitespace-nowrap shrink-0"
          >
            <Database className="w-4 h-4 text-[#00A99D]" />
            <span>Cargar base de ejemplo</span>
          </button>
        </div>

        {errorMsg && (
          <div className="flex items-start gap-2.5 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0 text-red-600" />
            <span>{errorMsg}</span>
          </div>
        )}

        {fileName && (
          <div className="flex items-center justify-between gap-4 rounded-lg border border-emerald-200 bg-emerald-50/70 px-4 py-3 text-sm text-emerald-900">
            <div className="flex items-center gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-[#65A30D] shrink-0" />
              <span>
                Base activa: <strong className="font-semibold">{fileName}</strong>
              </span>
            </div>
            <span className="font-mono text-xs tabular-nums text-emerald-800">
              {rowCount.toLocaleString('es-CO')} filas · {colCount} columnas
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* PASO 2: Vista previa y selección del sector económico                       */
/* -------------------------------------------------------------------------- */
interface Step2Props {
  rows: DataRow[];
  columns: string[];
  sector: string;
  customSector: string;
  onSectorChange: (s: string) => void;
  onCustomSectorChange: (s: string) => void;
}

export function Step2PreviewSector({
  rows,
  columns,
  sector,
  customSector,
  onSectorChange,
  onCustomSectorChange,
}: Step2Props) {
  const previewRows = rows.slice(0, 5);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="text-xs font-medium text-slate-500">Total de registros</div>
          <div className="text-2xl font-bold text-slate-900 tabular-nums mt-1">
            {rows.length.toLocaleString('es-CO')}
          </div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="text-xs font-medium text-slate-500">Total de columnas</div>
          <div className="text-2xl font-bold text-slate-900 tabular-nums mt-1">{columns.length}</div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="text-xs font-medium text-slate-500">Dimensiones generales</div>
          <div className="text-2xl font-bold text-slate-900 tabular-nums mt-1">
            {rows.length.toLocaleString('es-CO')} × {columns.length}
          </div>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
        <h3 className="text-base font-semibold text-slate-900">Vista previa (primeras 5 filas)</h3>
        <div className="overflow-x-auto border border-slate-200 rounded-lg">
          <table className="min-w-full text-xs text-left">
            <thead className="bg-[#0033A0] text-white">
              <tr>
                {columns.map((col) => (
                  <th key={col} className="px-3.5 py-2.5 font-semibold whitespace-nowrap">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white">
              {previewRows.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-50">
                  {columns.map((col) => {
                    const val = row[col];
                    const isNum = typeof val === 'number';
                    return (
                      <td
                        key={col}
                        className={`px-3.5 py-2 whitespace-nowrap ${
                          isNum ? 'font-mono tabular-nums text-slate-800' : 'text-slate-700'
                        }`}
                      >
                        {val === null || val === undefined ? (
                          <span className="text-slate-400 italic">NaN</span>
                        ) : (
                          String(val)
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
        <div>
          <h3 className="text-base font-semibold text-slate-900">Sector económico (obligatorio)</h3>
          <p className="text-sm text-slate-600 mt-1">
            Esta selección contextualiza los perfiles de segmentos y las recomendaciones estratégicas de los cinco pilares.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1.5">
              Seleccione el sector económico
            </label>
            <select
              value={sector}
              onChange={(e) => onSectorChange(e.target.value)}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-[#0033A0]"
            >
              {ECONOMIC_SECTORS.map((sec) => (
                <option key={sec} value={sec}>
                  {sec}
                </option>
              ))}
            </select>
          </div>

          {sector === 'Otro' && (
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Especifique el sector económico
              </label>
              <input
                type="text"
                value={customSector}
                onChange={(e) => onCustomSectorChange(e.target.value)}
                placeholder="Ej. Agroindustria, Turismo y hotelería..."
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-[#0033A0]"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* PASO 3: Inspección y clasificación de variables                             */
/* -------------------------------------------------------------------------- */
interface Step3Props {
  inspection: ColumnInspection[];
  idCols: string[];
  numericCols: string[];
  onUpdateSelection: (ids: string[], nums: string[]) => void;
}

export function Step3VariableInspection({
  inspection,
  idCols,
  numericCols,
  onUpdateSelection,
}: Step3Props) {
  const toggleId = (col: string) => {
    const nextIds = idCols.includes(col) ? idCols.filter((c) => c !== col) : [...idCols, col];
    const nextNums = numericCols.filter((c) => !nextIds.includes(c));
    onUpdateSelection(nextIds, nextNums);
  };

  const toggleNumeric = (col: string) => {
    if (idCols.includes(col)) return;
    const nextNums = numericCols.includes(col)
      ? numericCols.filter((c) => c !== col)
      : [...numericCols, col];
    onUpdateSelection(idCols, nextNums);
  };

  const zeroVarianceSelected = inspection
    .filter((c) => numericCols.includes(c.variable) && c.zeroVariance)
    .map((c) => c.variable);

  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
        <div>
          <h2 className="text-base font-semibold text-slate-900">Tabla de inspección y clasificación de variables</h2>
          <p className="text-sm text-slate-600 mt-1">
            Marque los identificadores (IDs) que deben excluirse y seleccione al menos dos variables numéricas con variabilidad para el clustering.
          </p>
        </div>

        <div className="overflow-x-auto border border-slate-200 rounded-lg">
          <table className="min-w-full text-sm text-left">
            <thead className="bg-[#0033A0] text-white text-xs">
              <tr>
                <th className="px-4 py-3 font-semibold">Variable</th>
                <th className="px-4 py-3 font-semibold">Tipo de dato</th>
                <th className="px-4 py-3 font-semibold text-right">Valores nulos</th>
                <th className="px-4 py-3 font-semibold text-right">Valores únicos</th>
                <th className="px-4 py-3 font-semibold text-center">Excluir como ID</th>
                <th className="px-4 py-3 font-semibold text-center">Usar en Clustering</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white">
              {inspection.map((item) => {
                const isId = idCols.includes(item.variable);
                const isNumSelected = numericCols.includes(item.variable);
                const canBeModelFeature = item.dataType === 'Numérica' && !isId && !item.zeroVariance;

                return (
                  <tr key={item.variable} className="hover:bg-slate-50">
                    <td className="px-4 py-2.5 font-medium text-slate-900">{item.variable}</td>
                    <td className="px-4 py-2.5 text-slate-600">
                      {item.dataType}
                      {item.zeroVariance && (
                        <span className="ml-2 text-xs text-amber-700 font-medium">(Sin variabilidad)</span>
                      )}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono tabular-nums text-slate-700">
                      {item.nullCount.toLocaleString('es-CO')}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono tabular-nums text-slate-700">
                      {item.uniqueCount.toLocaleString('es-CO')}
                    </td>
                    <td className="px-4 py-2.5 text-center">
                      <input
                        type="checkbox"
                        checked={isId}
                        onChange={() => toggleId(item.variable)}
                        aria-label={`Marcar ${item.variable} como identificador`}
                        className="h-4 w-4 accent-[#0033A0] cursor-pointer"
                      />
                    </td>
                    <td className="px-4 py-2.5 text-center">
                      <input
                        type="checkbox"
                        checked={isNumSelected}
                        disabled={!canBeModelFeature}
                        onChange={() => toggleNumeric(item.variable)}
                        aria-label={`Usar ${item.variable} en clustering`}
                        className="h-4 w-4 accent-[#65A30D] cursor-pointer disabled:opacity-35 disabled:cursor-not-allowed"
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {numericCols.length < 2 && (
          <div className="flex items-center gap-2.5 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
            <AlertTriangle className="w-4 h-4 text-red-600 shrink-0" />
            <span>Debe seleccionar al menos dos (2) variables numéricas válidas para avanzar al modelado.</span>
          </div>
        )}

        {zeroVarianceSelected.length > 0 && (
          <div className="flex items-center gap-2.5 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
            <span>Excluya las variables sin variabilidad: {zeroVarianceSelected.join(', ')}.</span>
          </div>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* PASO 4: Limpieza y diagnóstico de datos faltantes                           */
/* -------------------------------------------------------------------------- */
interface Step4Props {
  originalCount: number;
  currentCount: number;
  duplicatesDetected: number;
  duplicatesRemoved: number;
  inspection: ColumnInspection[];
  onRemoveDuplicates: () => void;
}

export function Step4CleaningDuplicates({
  originalCount,
  currentCount,
  duplicatesDetected,
  duplicatesRemoved,
  inspection,
  onRemoveDuplicates,
}: Step4Props) {
  const sorted = useMemo(
    () => [...inspection].sort((a, b) => b.nullCount - a.nullCount),
    [inspection]
  );

  const chartData = useMemo(
    () => [
      {
        x: sorted.map((s) => s.variable),
        y: sorted.map((s) => s.nullPct),
        text: sorted.map((s) => `${s.nullCount} nulos`),
        type: 'bar',
        marker: { color: '#0033A0' },
      },
    ],
    [sorted]
  );

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="text-xs font-medium text-slate-500">Registros antes de limpieza</div>
          <div className="text-2xl font-bold text-slate-900 tabular-nums mt-1">
            {originalCount.toLocaleString('es-CO')}
          </div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="text-xs font-medium text-slate-500">Registros actuales</div>
          <div className="text-2xl font-bold text-slate-900 tabular-nums mt-1">
            {currentCount.toLocaleString('es-CO')}
          </div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="text-xs font-medium text-slate-500">Duplicados detectados</div>
          <div className="text-2xl font-bold text-slate-900 tabular-nums mt-1">
            {duplicatesDetected.toLocaleString('es-CO')}
          </div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="text-xs font-medium text-slate-500">Duplicados eliminados</div>
          <div className="text-2xl font-bold text-[#65A30D] tabular-nums mt-1">
            {duplicatesRemoved.toLocaleString('es-CO')}
          </div>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h3 className="text-base font-semibold text-slate-900">Control de registros duplicados</h3>
          <p className="text-sm text-slate-600 mt-0.5">
            {duplicatesDetected > 0
              ? `Se identificaron ${duplicatesDetected} filas exactamente duplicadas en la base.`
              : 'La base actual no contiene registros duplicados.'}
          </p>
        </div>
        <button
          type="button"
          disabled={duplicatesDetected === 0}
          onClick={onRemoveDuplicates}
          className="inline-flex items-center justify-center gap-2 rounded-lg bg-[#0033A0] hover:bg-[#002880] disabled:bg-slate-200 disabled:text-slate-500 px-4 py-2.5 text-sm font-medium text-white transition-colors whitespace-nowrap shrink-0"
        >
          <Trash2 className="w-4 h-4" />
          <span>Eliminar duplicados</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-xl p-6">
          <h3 className="text-base font-semibold text-slate-900 mb-3">Tabla de diagnóstico de faltantes</h3>
          <div className="overflow-x-auto border border-slate-200 rounded-lg">
            <table className="min-w-full text-sm text-left">
              <thead className="bg-slate-100 text-slate-700 text-xs">
                <tr>
                  <th className="px-4 py-2.5 font-semibold">Variable</th>
                  <th className="px-4 py-2.5 font-semibold text-right">Nulos</th>
                  <th className="px-4 py-2.5 font-semibold text-right">Faltante (%)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {sorted.map((row) => (
                  <tr key={row.variable}>
                    <td className="px-4 py-2 font-medium text-slate-800">{row.variable}</td>
                    <td className="px-4 py-2 text-right font-mono tabular-nums">{row.nullCount}</td>
                    <td className="px-4 py-2 text-right font-mono tabular-nums">{row.nullPct.toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-6">
          <PlotlyChart
            data={chartData}
            layout={{
              title: { text: 'Porcentaje de datos faltantes por columna', font: { size: 15 } },
              yaxis: { title: 'Porcentaje (%)' },
              margin: { l: 50, r: 20, t: 50, b: 80 },
            }}
            className="w-full h-[320px]"
          />
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* PASO 5: Imputación de datos                                                 */
/* -------------------------------------------------------------------------- */
interface Step5Props {
  inspection: ColumnInspection[];
  strategies: Record<string, ImputationStrategy>;
  dropRowsWithNulls: boolean;
  summary: string[];
  onStrategyChange: (col: string, strategy: ImputationStrategy) => void;
  onDropRowsToggle: (drop: boolean) => void;
  onApplyImputation: () => void;
}

export function Step5Imputation({
  inspection,
  strategies,
  dropRowsWithNulls,
  summary,
  onStrategyChange,
  onDropRowsToggle,
  onApplyImputation,
}: Step5Props) {
  const colsWithNulls = inspection.filter((c) => c.nullCount > 0);

  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-5">
        <div>
          <h2 className="text-base font-semibold text-slate-900">Tratamiento de valores faltantes</h2>
          <p className="text-sm text-slate-600 mt-1">
            Las transformaciones se aplican sobre una copia de trabajo; la base original cargada se mantiene intacta.
          </p>
        </div>

        {colsWithNulls.length === 0 ? (
          <div className="flex items-center gap-2.5 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">
            <CheckCircle2 className="w-4 h-4 text-[#65A30D]" />
            <span>No existen valores faltantes pendientes en la base activa. Puede continuar al siguiente paso.</span>
          </div>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={() => onDropRowsToggle(false)}
                className={`px-4 py-2 rounded-lg text-sm font-medium border transition-colors ${
                  !dropRowsWithNulls
                    ? 'bg-[#0033A0] text-white border-[#0033A0]'
                    : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'
                }`}
              >
                Imputar por columna
              </button>
              <button
                type="button"
                onClick={() => onDropRowsToggle(true)}
                className={`px-4 py-2 rounded-lg text-sm font-medium border transition-colors ${
                  dropRowsWithNulls
                    ? 'bg-[#0033A0] text-white border-[#0033A0]'
                    : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'
                }`}
              >
                Eliminar registros con nulos
              </button>
            </div>

            {!dropRowsWithNulls && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {colsWithNulls.map((colInfo) => {
                  const isNum = colInfo.dataType === 'Numérica';
                  const currentVal =
                    strategies[colInfo.variable] || (isNum ? 'Mediana' : 'Moda');
                  return (
                    <div key={colInfo.variable} className="border border-slate-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-semibold text-slate-900">{colInfo.variable}</span>
                        <span className="text-xs font-mono tabular-nums text-slate-500">
                          {colInfo.dataType} · {colInfo.nullCount} nulos
                        </span>
                      </div>
                      <select
                        value={currentVal}
                        onChange={(e) =>
                          onStrategyChange(colInfo.variable, e.target.value as ImputationStrategy)
                        }
                        className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-[#0033A0]"
                      >
                        {isNum ? (
                          <>
                            <option value="Media">Media aritmética</option>
                            <option value="Mediana">Mediana</option>
                            <option value="Cero">Sustitución por cero (0)</option>
                            <option value="KNNImputer">KNNImputer (5 vecinos cercanos)</option>
                          </>
                        ) : (
                          <>
                            <option value="Moda">Moda (valor más frecuente)</option>
                            <option value="Desconocido">Sustitución por "Desconocido"</option>
                          </>
                        )}
                      </select>
                    </div>
                  );
                })}
              </div>
            )}

            <button
              type="button"
              onClick={onApplyImputation}
              className="inline-flex items-center gap-2 rounded-lg bg-[#0033A0] hover:bg-[#002880] px-5 py-2.5 text-sm font-medium text-white transition-colors"
            >
              Aplicar imputación
            </button>
          </>
        )}

        {summary.length > 0 && (
          <div className="border-t border-slate-200 pt-4">
            <h3 className="text-sm font-semibold text-slate-900 mb-2">Resumen de cambios aplicados</h3>
            <ul className="space-y-1 text-sm text-slate-700 list-disc pl-5">
              {summary.map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* PASO 6: Análisis descriptivo y reglas de negocio                            */
/* -------------------------------------------------------------------------- */
interface Step6Props {
  rowsPreRules: DataRow[];
  rowsAfterRules: DataRow[];
  numericCols: string[];
  stats: DescriptiveStatsRow[];
  rules: Record<string, BusinessRuleRange>;
  ruleSummary: string[];
  onRuleChange: (col: string, field: 'min' | 'max', value: number) => void;
  onApplyRules: () => void;
}

export function Step6DescriptiveBusinessRules({
  rowsPreRules,
  rowsAfterRules,
  numericCols,
  stats,
  rules,
  ruleSummary,
  onRuleChange,
  onApplyRules,
}: Step6Props) {
  const [selectedVar, setSelectedVar] = useState<string>(numericCols[0] || '');
  const activeVar = numericCols.includes(selectedVar) ? selectedVar : numericCols[0] || '';

  const activeValues = useMemo(
    () =>
      rowsPreRules
        .map((r) => r[activeVar])
        .filter((v): v is number => typeof v === 'number' && Number.isFinite(v)),
    [rowsPreRules, activeVar]
  );

  const histData = useMemo(
    () => [
      {
        x: activeValues,
        type: 'histogram',
        nbinsx: 28,
        marker: { color: '#0033A0', line: { color: '#FFFFFF', width: 0.6 } },
      },
    ],
    [activeValues]
  );

  const boxData = useMemo(
    () => [
      {
        y: activeValues,
        type: 'box',
        boxpoints: 'outliers',
        name: activeVar,
        marker: { color: '#00A99D' },
      },
    ],
    [activeValues, activeVar]
  );

  const fmt = (n: number) => n.toLocaleString('es-CO', { maximumFractionDigits: 2 });

  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
        <h2 className="text-base font-semibold text-slate-900">Estadísticas descriptivas (describe)</h2>
        <div className="overflow-x-auto border border-slate-200 rounded-lg">
          <table className="min-w-full text-xs text-right">
            <thead className="bg-[#0033A0] text-white">
              <tr>
                <th className="px-3.5 py-2.5 text-left font-semibold">Variable</th>
                <th className="px-3.5 py-2.5 font-semibold">Conteo</th>
                <th className="px-3.5 py-2.5 font-semibold">Media</th>
                <th className="px-3.5 py-2.5 font-semibold">Desv. Est.</th>
                <th className="px-3.5 py-2.5 font-semibold">Mín</th>
                <th className="px-3.5 py-2.5 font-semibold">25%</th>
                <th className="px-3.5 py-2.5 font-semibold">50%</th>
                <th className="px-3.5 py-2.5 font-semibold">75%</th>
                <th className="px-3.5 py-2.5 font-semibold">Máx</th>
                <th className="px-3.5 py-2.5 font-semibold">Atípicos IQR</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 font-mono tabular-nums">
              {stats.map((s) => (
                <tr key={s.variable} className="hover:bg-slate-50">
                  <td className="px-3.5 py-2 text-left font-sans font-semibold text-slate-900">
                    {s.variable}
                  </td>
                  <td className="px-3.5 py-2">{s.count}</td>
                  <td className="px-3.5 py-2">{fmt(s.mean)}</td>
                  <td className="px-3.5 py-2">{fmt(s.std)}</td>
                  <td className="px-3.5 py-2">{fmt(s.min)}</td>
                  <td className="px-3.5 py-2">{fmt(s.q25)}</td>
                  <td className="px-3.5 py-2">{fmt(s.median)}</td>
                  <td className="px-3.5 py-2">{fmt(s.q75)}</td>
                  <td className="px-3.5 py-2">{fmt(s.max)}</td>
                  <td className="px-3.5 py-2 text-amber-700 font-semibold">{s.iqrOutliers}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <h3 className="text-base font-semibold text-slate-900">Distribución y detección visual de atípicos</h3>
          <select
            value={activeVar}
            onChange={(e) => setSelectedVar(e.target.value)}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-[#0033A0]"
          >
            {numericCols.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <PlotlyChart
            data={histData}
            layout={{
              title: { text: `Histograma: ${activeVar}`, font: { size: 14 } },
              xaxis: { title: activeVar },
              yaxis: { title: 'Frecuencia' },
              margin: { l: 50, r: 20, t: 45, b: 50 },
            }}
            className="w-full h-[320px]"
          />
          <PlotlyChart
            data={boxData}
            layout={{
              title: { text: `Boxplot: ${activeVar}`, font: { size: 14 } },
              yaxis: { title: activeVar },
              margin: { l: 60, r: 20, t: 45, b: 40 },
            }}
            className="w-full h-[320px]"
          />
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold text-slate-900">Reglas de negocio (límites mínimos y máximos)</h3>
            <p className="text-sm text-slate-600">
              Excluya registros inválidos (ej. edades improbables, montos negativos) antes de entrenar el modelo.
            </p>
          </div>
          <div className="text-sm font-mono tabular-nums text-slate-800">
            Disponibles para modelar: <strong>{rowsAfterRules.length.toLocaleString('es-CO')}</strong> /{' '}
            {rowsPreRules.length.toLocaleString('es-CO')}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {numericCols.map((col) => {
            const st = stats.find((x) => x.variable === col);
            const r = rules[col] || { min: st?.min ?? 0, max: st?.max ?? 0 };
            return (
              <div key={col} className="border border-slate-200 rounded-lg p-4 space-y-2">
                <div className="text-sm font-semibold text-slate-900">{col}</div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs text-slate-500 mb-1">Mínimo permitido</label>
                    <input
                      type="number"
                      step="any"
                      value={r.min}
                      onChange={(e) => onRuleChange(col, 'min', Number(e.target.value))}
                      className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 text-sm font-mono tabular-nums"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-slate-500 mb-1">Máximo permitido</label>
                    <input
                      type="number"
                      step="any"
                      value={r.max}
                      onChange={(e) => onRuleChange(col, 'max', Number(e.target.value))}
                      className="w-full rounded-md border border-slate-300 px-2.5 py-1.5 text-sm font-mono tabular-nums"
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <button
            type="button"
            onClick={onApplyRules}
            className="inline-flex items-center gap-2 rounded-lg bg-[#0033A0] hover:bg-[#002880] px-5 py-2.5 text-sm font-medium text-white transition-colors"
          >
            <Filter className="w-4 h-4" />
            <span>Aplicar reglas de negocio</span>
          </button>
        </div>

        {ruleSummary.length > 0 && (
          <div className="border-t border-slate-200 pt-3">
            <ul className="space-y-1 text-sm text-slate-700 list-disc pl-5">
              {ruleSummary.map((line, idx) => (
                <li key={idx}>{line}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
