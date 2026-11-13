import { useMemo, useState } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  Download,
  KeyRound,
  Menu,
  X,
  FileCode2,
} from 'lucide-react';
import streamlitAppRaw from '../app.py?raw';
import requirementsRaw from '../requirements.txt?raw';
import {
  Step1Upload,
  Step2PreviewSector,
  Step3VariableInspection,
  Step4CleaningDuplicates,
  Step5Imputation,
  Step6DescriptiveBusinessRules,
} from './components/DataPrepSteps';
import {
  Step7OptimalK,
  Step8AlgorithmSelection,
  Step9Execution,
  Step10CentroidsVisuals,
  Step11Profiling,
  Step12Strategies,
  Step13ExportReport,
} from './components/ModelingStrategySteps';
import {
  BusinessRuleRange,
  CentroidRow,
  ClusterMetrics,
  ClusterProfile,
  ClusteringAlgorithm,
  DataRow,
  DistanceMetric,
  EncodingOption,
  FivePillarsStrategy,
  ImputationStrategy,
  KMetricPoint,
  LinkageMethod,
  PCAProjection,
  STEP_DEFINITIONS,
  ScalingMethod,
  SeparatorOption,
} from './types/clustering';
import {
  applyBusinessRules,
  applyImputation,
  computeDescriptiveStats,
  countDuplicates,
  generateSampleDataset,
  inspectColumns,
  parseUploadedFile,
  removeDuplicateRows,
} from './utils/dataProcessing';
import {
  evaluateKRange,
  executeClusteringPipeline,
  extractNumericMatrix,
  fitTransformScaler,
  validateHierarchicalCombination,
} from './utils/mlEngine';
import {
  buildSelfContainedHtmlReport,
  generateOpenAIProfilesAndStrategies,
  generateSimulationProfilesAndStrategies,
} from './utils/commercialStrategy';

function triggerDownload(filename: string, content: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export default function App() {
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  // Configuración general y OpenAI (solo en memoria de sesión)
  const [projectName, setProjectName] = useState('Análisis de Segmentación Comercial');
  const [apiKey, setApiKey] = useState('');
  const [openAIModel, setOpenAIModel] = useState('gpt-4o-mini');

  // Paso 1: Archivo
  const [separator, setSeparator] = useState<SeparatorOption>(',');
  const [encoding, setEncoding] = useState<EncodingOption>('utf-8');
  const [fileName, setFileName] = useState<string | null>(null);
  const [rawRows, setRawRows] = useState<DataRow[]>([]);
  const [columns, setColumns] = useState<string[]>([]);

  // Paso 2: Sector económico
  const [sector, setSector] = useState<string>('Retail y comercio');
  const [customSector, setCustomSector] = useState<string>('');

  // Paso 3: Variables
  const [idCols, setIdCols] = useState<string[]>([]);
  const [numericCols, setNumericCols] = useState<string[]>([]);

  // Paso 4: Limpieza y duplicados
  const [cleanedRows, setCleanedRows] = useState<DataRow[]>([]);
  const [duplicatesRemoved, setDuplicatesRemoved] = useState<number>(0);

  // Paso 5: Imputación
  const [imputationStrategies, setImputationStrategies] = useState<Record<string, ImputationStrategy>>({});
  const [dropRowsWithNulls, setDropRowsWithNulls] = useState<boolean>(false);
  const [imputedRows, setImputedRows] = useState<DataRow[]>([]);
  const [imputationSummary, setImputationSummary] = useState<string[]>([]);

  // Paso 6: Reglas de negocio
  const [businessRules, setBusinessRules] = useState<Record<string, BusinessRuleRange>>({});
  const [filteredRows, setFilteredRows] = useState<DataRow[]>([]);
  const [ruleSummary, setRuleSummary] = useState<string[]>([]);

  // Paso 7: Número óptimo K
  const [scalingMethod, setScalingMethod] = useState<ScalingMethod>('StandardScaler');
  const [kEvaluation, setKEvaluation] = useState<{
    points: KMetricPoint[];
    bestSilhouetteK: number;
    elbowK: number;
  } | null>(null);
  const [selectedK, setSelectedK] = useState<number>(4);
  const [businessKNote, setBusinessKNote] = useState<string>('');

  // Paso 8: Algoritmo y distancia
  const [algorithm, setAlgorithm] = useState<ClusteringAlgorithm>('K-Means');
  const [metric, setMetric] = useState<DistanceMetric>('euclidean');
  const [linkage, setLinkage] = useState<LinkageMethod>('ward');
  const [nInit, setNInit] = useState<number>(10);
  const [randomState, setRandomState] = useState<number>(42);
  const [maxIter, setMaxIter] = useState<number>(300);
  const [tol, setTol] = useState<number>(0.0001);

  // Paso 9 & 10: Resultados del modelo
  const [clusterLabels, setClusterLabels] = useState<number[] | null>(null);
  const [clusterMetrics, setClusterMetrics] = useState<ClusterMetrics | null>(null);
  const [centroids, setCentroids] = useState<CentroidRow[]>([]);
  const [pcaProjection, setPcaProjection] = useState<PCAProjection | null>(null);

  // Paso 11 & 12: Perfilamiento y estrategias
  const [profiles, setProfiles] = useState<Record<number, ClusterProfile> | null>(null);
  const [strategies, setStrategies] = useState<Record<number, FivePillarsStrategy> | null>(null);
  const [aiModeUsed, setAiModeUsed] = useState<string>('Simulación / Plantilla');
  const [aiLoading, setAiLoading] = useState<boolean>(false);
  const [aiError, setAiError] = useState<string | null>(null);

  const effectiveSector = useMemo(() => {
    if (sector === 'Otro') {
      return customSector.trim() || 'Otro sector económico';
    }
    return sector;
  }, [sector, customSector]);

  // Tablas derivadas
  const rawInspection = useMemo(
    () => (rawRows.length ? inspectColumns(rawRows, columns) : []),
    [rawRows, columns]
  );

  const activeCleanRows = cleanedRows.length ? cleanedRows : rawRows;
  const cleanInspection = useMemo(
    () => (activeCleanRows.length ? inspectColumns(activeCleanRows, columns) : []),
    [activeCleanRows, columns]
  );

  const duplicatesDetected = useMemo(
    () => (activeCleanRows.length ? countDuplicates(activeCleanRows, columns) : 0),
    [activeCleanRows, columns]
  );

  const activeImputedRows = imputedRows.length ? imputedRows : activeCleanRows;
  const activeModeledRows = filteredRows.length ? filteredRows : activeImputedRows;

  const descriptiveStats = useMemo(
    () =>
      activeImputedRows.length && numericCols.length
        ? computeDescriptiveStats(activeImputedRows, numericCols)
        : [],
    [activeImputedRows, numericCols]
  );

  const labeledRows = useMemo(() => {
    if (!clusterLabels) return [];
    return activeModeledRows.map((row, idx) => ({
      ...row,
      Cluster: clusterLabels[idx],
    }));
  }, [activeModeledRows, clusterLabels]);

  // Invalidación controlada de pasos posteriores
  const invalidateFromStep = (fromStep: number) => {
    if (fromStep <= 1) {
      setCleanedRows([]);
      setDuplicatesRemoved(0);
      setIdCols([]);
      setNumericCols([]);
    }
    if (fromStep <= 4) {
      setImputedRows([]);
      setImputationSummary([]);
    }
    if (fromStep <= 5) {
      setFilteredRows([]);
      setBusinessRules({});
      setRuleSummary([]);
    }
    if (fromStep <= 6) {
      setKEvaluation(null);
    }
    if (fromStep <= 8) {
      setClusterLabels(null);
      setClusterMetrics(null);
      setCentroids([]);
      setPcaProjection(null);
    }
    if (fromStep <= 10) {
      setProfiles(null);
      setStrategies(null);
      setAiError(null);
    }
  };

  const initializeDataset = (rows: DataRow[], cols: string[], name: string) => {
    invalidateFromStep(1);
    setRawRows(rows);
    setColumns(cols);
    setFileName(name);
    setCleanedRows(rows.map((r) => ({ ...r })));

    const insp = inspectColumns(rows, cols);
    const detectedIds = insp
      .filter((c) => c.variable.toLowerCase().includes('id'))
      .map((c) => c.variable);
    const detectedNums = insp
      .filter((c) => c.dataType === 'Numérica' && !detectedIds.includes(c.variable) && !c.zeroVariance)
      .map((c) => c.variable);

    setIdCols(detectedIds);
    setNumericCols(detectedNums);
  };

  const handleFileUpload = async (file: File) => {
    const { rows, columns: cols } = await parseUploadedFile(file, separator, encoding);
    initializeDataset(rows, cols, file.name);
  };

  const handleLoadSample = () => {
    const { rows, columns: cols } = generateSampleDataset();
    initializeDataset(rows, cols, 'clientes_ejemplo_retail.csv');
  };

  const handleRemoveDuplicates = () => {
    const { cleaned, removed } = removeDuplicateRows(activeCleanRows, columns);
    setCleanedRows(cleaned);
    setDuplicatesRemoved((prev) => prev + removed);
    invalidateFromStep(4);
  };

  const handleApplyImputation = () => {
    const { imputed, summary } = applyImputation(
      activeCleanRows,
      columns,
      cleanInspection,
      imputationStrategies,
      dropRowsWithNulls
    );
    setImputedRows(imputed);
    setImputationSummary(summary);
    setFilteredRows(imputed);
    invalidateFromStep(6);

    const stats = computeDescriptiveStats(imputed, numericCols);
    const initialRules: Record<string, BusinessRuleRange> = {};
    for (const s of stats) {
      initialRules[s.variable] = { min: s.min, max: s.max };
    }
    setBusinessRules(initialRules);
  };

  const handleApplyBusinessRules = () => {
    const effectiveRules: Record<string, BusinessRuleRange> = {};
    for (const st of descriptiveStats) {
      effectiveRules[st.variable] = businessRules[st.variable] || { min: st.min, max: st.max };
    }
    const { filtered, ruleSummary: summary } = applyBusinessRules(activeImputedRows, effectiveRules);
    if (filtered.length < 4) {
      setRuleSummary([
        'Advertencia: las reglas definidas dejan menos de 4 registros. Ajuste los límites antes de continuar.',
      ]);
      return;
    }
    setFilteredRows(filtered);
    setRuleSummary(summary);
    invalidateFromStep(6);
  };

  const computeKMetricsNow = (method: ScalingMethod = scalingMethod) => {
    const rowsToUse = activeModeledRows;
    const rawMat = extractNumericMatrix(rowsToUse, numericCols);
    const { scaled } = fitTransformScaler(rawMat, method);
    const res = evaluateKRange(scaled, 10);
    setKEvaluation(res);
    if (selectedK > (res.points[res.points.length - 1]?.k || 10)) {
      setSelectedK(res.bestSilhouetteK);
    }
  };

  const handleRunClustering = () => {
    const res = executeClusteringPipeline({
      rows: activeModeledRows,
      numericCols,
      scalingMethod,
      algorithm,
      k: selectedK,
      metric: algorithm === 'K-Means' ? 'euclidean' : metric,
      linkage,
      nInit,
      randomState,
      maxIter,
      tol,
    });
    setClusterLabels(res.labels);
    setClusterMetrics(res.metrics);
    setCentroids(res.centroids);
    setPcaProjection(res.pca);
    invalidateFromStep(10);
  };

  const handleGenerateProfiles = async () => {
    setAiError(null);
    if (apiKey.trim()) {
      setAiLoading(true);
      try {
        const res = await generateOpenAIProfilesAndStrategies(
          apiKey.trim(),
          openAIModel,
          centroids,
          numericCols,
          effectiveSector
        );
        setProfiles(res.profiles);
        setStrategies(res.strategies);
        setAiModeUsed(`Inteligencia Artificial (${openAIModel})`);
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Error al conectar con OpenAI.';
        setAiError(`${msg} Se aplicó el Modo Simulación / Plantilla como respaldo.`);
        const sim = generateSimulationProfilesAndStrategies(
          centroids,
          numericCols,
          activeModeledRows,
          effectiveSector
        );
        setProfiles(sim.profiles);
        setStrategies(sim.strategies);
        setAiModeUsed('Simulación / Plantilla (respaldo)');
      } finally {
        setAiLoading(false);
      }
    } else {
      const sim = generateSimulationProfilesAndStrategies(
        centroids,
        numericCols,
        activeModeledRows,
        effectiveSector
      );
      setProfiles(sim.profiles);
      setStrategies(sim.strategies);
      setAiModeUsed('Simulación / Plantilla');
    }
  };

  const handleDownloadHtmlReport = () => {
    if (!clusterLabels || !clusterMetrics || !pcaProjection || !profiles || !strategies) return;
    const html = buildSelfContainedHtmlReport({
      projectName,
      sector: effectiveSector,
      fileName: fileName || 'base_datos',
      originalRowCount: rawRows.length,
      originalColCount: columns.length,
      duplicatesRemoved,
      modeledRowCount: activeModeledRows.length,
      idCols,
      numericCols,
      imputationSummary: imputationSummary.length ? imputationSummary : ['Sin imputaciones necesarias.'],
      businessRuleSummary: ruleSummary.length ? ruleSummary : ['Sin exclusiones por reglas de negocio.'],
      scalingMethod,
      algorithm,
      k: selectedK,
      metric: algorithm === 'K-Means' ? 'euclidean' : metric,
      linkage,
      businessKNote,
      metrics: clusterMetrics,
      centroids,
      labels: clusterLabels,
      pca: pcaProjection,
      profiles,
      strategies,
      aiModeLabel: aiModeUsed,
    });
    triggerDownload('informe_ejecutivo_clustering.html', html, 'text/html;charset=utf-8');
  };

  // Validación de disponibilidad de pasos
  const isStepAvailable = (stepId: number): boolean => {
    if (stepId === 1) return true;
    if (!rawRows.length) return false;
    if (stepId <= 3) return true;
    const hasZeroVar = rawInspection.some(
      (c) => numericCols.includes(c.variable) && c.zeroVariance
    );
    if (numericCols.length < 2 || hasZeroVar) return false;
    if (stepId <= 5) return true;
    const missingInSelected = activeImputedRows.some((row) =>
      numericCols.some((c) => row[c] === null || row[c] === undefined || typeof row[c] !== 'number')
    );
    if (missingInSelected && stepId >= 6) return false;
    if (stepId <= 9) return true;
    if (!clusterLabels || !pcaProjection) return false;
    if (stepId <= 11) return true;
    if (!profiles || !strategies) return false;
    return true;
  };

  const canGoNext = (): boolean => {
    if (currentStep === 2 && sector === 'Otro' && !customSector.trim()) return false;
    if (currentStep === 8 && algorithm !== 'K-Means') {
      return validateHierarchicalCombination(metric, linkage).valid;
    }
    return currentStep < 13 && isStepAvailable(currentStep + 1);
  };

  const goToStep = (step: number) => {
    if (step === 5 && imputedRows.length === 0) {
      setImputedRows(activeCleanRows.map((r) => ({ ...r })));
    }
    if (step === 6 && Object.keys(businessRules).length === 0 && descriptiveStats.length > 0) {
      const init: Record<string, BusinessRuleRange> = {};
      for (const s of descriptiveStats) init[s.variable] = { min: s.min, max: s.max };
      setBusinessRules(init);
      if (filteredRows.length === 0) setFilteredRows(activeImputedRows);
    }
    if (step === 7 && !kEvaluation) {
      computeKMetricsNow(scalingMethod);
    }
    setCurrentStep(step);
    setMobileNavOpen(false);
  };

  const activeDef = STEP_DEFINITIONS.find((s) => s.id === currentStep) || STEP_DEFINITIONS[0];

  const sidebarContent = (
    <div className="flex flex-col h-full bg-white border-r border-slate-200">
      <div className="p-5 border-b border-slate-200">
        <div className="flex items-center gap-3">
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none" aria-hidden="true">
            <rect width="32" height="32" rx="8" fill="#0033A0" />
            <circle cx="11" cy="12" r="3.5" fill="#FCD116" />
            <circle cx="21" cy="11" r="3.5" fill="#78BE20" />
            <circle cx="16" cy="21" r="4" fill="#00B5AD" />
          </svg>
          <div className="min-w-0">
            <div className="font-display font-bold text-sm text-slate-900 leading-tight">
              Clustering &amp; Strategy Lab
            </div>
            <div className="text-xs text-slate-500 truncate">Analítica Comercial</div>
          </div>
        </div>
        <div className="flex h-1 w-full rounded-full overflow-hidden mt-4">
          <div className="flex-1 bg-[#78BE20]" />
          <div className="flex-1 bg-[#FCD116]" />
          <div className="flex-1 bg-[#00B5AD]" />
          <div className="flex-1 bg-[#0033A0]" />
        </div>
      </div>

      <div className="p-4 border-b border-slate-200 space-y-3">
        <div>
          <label className="block text-xs font-semibold text-slate-600 mb-1">Nombre del proyecto</label>
          <input
            type="text"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            className="w-full rounded-lg border border-slate-300 px-2.5 py-1.5 text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-[#0033A0]"
          />
        </div>

        <div>
          <label className="flex items-center gap-1.5 text-xs font-semibold text-slate-600 mb-1">
            <KeyRound className="w-3.5 h-3.5 text-[#0033A0]" />
            <span>OpenAI API Key (Opcional)</span>
          </label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-... (vacío = Simulación)"
            className="w-full rounded-lg border border-slate-300 px-2.5 py-1.5 text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-[#0033A0]"
          />
        </div>

        <div>
          <label className="block text-xs font-semibold text-slate-600 mb-1">Modelo OpenAI</label>
          <select
            value={openAIModel}
            onChange={(e) => setOpenAIModel(e.target.value)}
            className="w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-900"
          >
            <option value="gpt-4o-mini">gpt-4o-mini</option>
            <option value="gpt-4o">gpt-4o</option>
            <option value="gpt-4.1-mini">gpt-4.1-mini</option>
          </select>
        </div>
      </div>

      <div className="px-4 py-3 border-b border-slate-200">
        <div className="flex items-center justify-between text-xs font-medium text-slate-600 mb-1.5">
          <span>Progreso</span>
          <span className="font-mono tabular-nums text-[#0033A0] font-semibold">
            Paso {currentStep} / 13
          </span>
        </div>
        <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-[#0033A0] transition-all duration-200"
            style={{ width: `${(currentStep / 13) * 100}%` }}
          />
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto p-3 space-y-1">
        {STEP_DEFINITIONS.map((step) => {
          const enabled = isStepAvailable(step.id);
          const active = currentStep === step.id;
          return (
            <button
              key={step.id}
              type="button"
              disabled={!enabled}
              onClick={() => goToStep(step.id)}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-medium text-left transition-colors ${
                active
                  ? 'bg-[#0033A0] text-white'
                  : enabled
                    ? 'text-slate-700 hover:bg-slate-100'
                    : 'text-slate-400 cursor-not-allowed'
              }`}
            >
              <span className="font-mono tabular-nums w-5 shrink-0">{step.id}.</span>
              <span className="truncate">{step.short}</span>
            </button>
          );
        })}
      </nav>

      <div className="p-4 border-t border-slate-200 space-y-2">
        <div className="text-xs font-semibold text-slate-700">Código Python / Streamlit</div>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => triggerDownload('app.py', streamlitAppRaw, 'text/x-python;charset=utf-8')}
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 px-2.5 py-2 text-xs font-medium text-slate-700"
          >
            <FileCode2 className="w-3.5 h-3.5 text-[#0033A0]" />
            <span>app.py</span>
          </button>
          <button
            type="button"
            onClick={() => triggerDownload('requirements.txt', requirementsRaw, 'text/plain;charset=utf-8')}
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 px-2.5 py-2 text-xs font-medium text-slate-700"
          >
            <Download className="w-3.5 h-3.5 text-[#00A99D]" />
            <span>reqs.txt</span>
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen flex bg-slate-50 text-slate-900">
      {/* Sidebar Desktop */}
      <aside className="hidden lg:block w-[272px] shrink-0 sticky top-0 h-screen">
        {sidebarContent}
      </aside>

      {/* Drawer Mobile */}
      {mobileNavOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div className="w-[280px] h-full shadow-xl">{sidebarContent}</div>
          <div className="flex-1 bg-slate-900/40" onClick={() => setMobileNavOpen(false)} />
        </div>
      )}

      {/* Main Workspace */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Bar Contract: Brand/Title — Quiet Metadata — 1 Primary Action */}
        <header className="sticky top-0 z-30 bg-white/95 backdrop-blur border-b border-slate-200 px-6 py-3.5 flex items-center justify-between gap-6">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={() => setMobileNavOpen(!mobileNavOpen)}
              className="lg:hidden p-2 rounded-lg border border-slate-200 text-slate-700"
              aria-label="Abrir navegación"
            >
              {mobileNavOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
            </button>
            <h1 className="text-base sm:text-lg font-semibold text-slate-900 truncate">
              {activeDef.title}
            </h1>
          </div>

          <div className="hidden md:flex items-center gap-2 text-xs text-slate-500 whitespace-nowrap shrink-0">
            <span>{effectiveSector}</span>
            <span aria-hidden="true">·</span>
            <span className="font-mono tabular-nums">
              {rawRows.length ? `${activeModeledRows.length.toLocaleString('es-CO')} registros` : 'Sin archivo'}
            </span>
            {numericCols.length > 0 && (
              <>
                <span aria-hidden="true">·</span>
                <span className="font-mono tabular-nums">{numericCols.length} variables</span>
              </>
            )}
          </div>

          <div className="shrink-0">
            {currentStep < 13 ? (
              <button
                type="button"
                disabled={!canGoNext()}
                onClick={() => goToStep(currentStep + 1)}
                className="inline-flex items-center gap-1.5 rounded-lg bg-[#0033A0] hover:bg-[#002880] disabled:bg-slate-200 disabled:text-slate-500 px-4 py-2 text-xs font-semibold text-white transition-colors whitespace-nowrap shrink-0"
              >
                <span>Siguiente paso</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            ) : (
              <button
                type="button"
                onClick={handleDownloadHtmlReport}
                className="inline-flex items-center gap-1.5 rounded-lg bg-[#0033A0] hover:bg-[#002880] px-4 py-2 text-xs font-semibold text-white transition-colors whitespace-nowrap shrink-0"
              >
                <Download className="w-4 h-4" />
                <span>Descargar HTML</span>
              </button>
            )}
          </div>
        </header>

        {/* Main Content Viewport */}
        <main className="flex-1 p-6 max-w-6xl w-full mx-auto">
          {currentStep === 1 && (
            <Step1Upload
              fileName={fileName}
              rowCount={rawRows.length}
              colCount={columns.length}
              separator={separator}
              encoding={encoding}
              onSeparatorChange={setSeparator}
              onEncodingChange={setEncoding}
              onFileUpload={handleFileUpload}
              onLoadSample={handleLoadSample}
            />
          )}

          {currentStep === 2 && (
            <Step2PreviewSector
              rows={rawRows}
              columns={columns}
              sector={sector}
              customSector={customSector}
              onSectorChange={setSector}
              onCustomSectorChange={setCustomSector}
            />
          )}

          {currentStep === 3 && (
            <Step3VariableInspection
              inspection={rawInspection}
              idCols={idCols}
              numericCols={numericCols}
              onUpdateSelection={(nextIds, nextNums) => {
                setIdCols(nextIds);
                setNumericCols(nextNums);
                invalidateFromStep(3);
              }}
            />
          )}

          {currentStep === 4 && (
            <Step4CleaningDuplicates
              originalCount={rawRows.length}
              currentCount={activeCleanRows.length}
              duplicatesDetected={duplicatesDetected}
              duplicatesRemoved={duplicatesRemoved}
              inspection={cleanInspection}
              onRemoveDuplicates={handleRemoveDuplicates}
            />
          )}

          {currentStep === 5 && (
            <Step5Imputation
              inspection={cleanInspection}
              strategies={imputationStrategies}
              dropRowsWithNulls={dropRowsWithNulls}
              summary={imputationSummary}
              onStrategyChange={(col, strat) =>
                setImputationStrategies((prev) => ({ ...prev, [col]: strat }))
              }
              onDropRowsToggle={setDropRowsWithNulls}
              onApplyImputation={handleApplyImputation}
            />
          )}

          {currentStep === 6 && (
            <Step6DescriptiveBusinessRules
              rowsPreRules={activeImputedRows}
              rowsAfterRules={activeModeledRows}
              numericCols={numericCols}
              stats={descriptiveStats}
              rules={businessRules}
              ruleSummary={ruleSummary}
              onRuleChange={(col, field, value) =>
                setBusinessRules((prev) => {
                  const stat = descriptiveStats.find((s) => s.variable === col);
                  const current = prev[col] || { min: stat?.min ?? 0, max: stat?.max ?? 0 };
                  return { ...prev, [col]: { ...current, [field]: value } };
                })
              }
              onApplyRules={handleApplyBusinessRules}
            />
          )}

          {currentStep === 7 && kEvaluation && (
            <Step7OptimalK
              scalingMethod={scalingMethod}
              kPoints={kEvaluation.points}
              bestSilhouetteK={kEvaluation.bestSilhouetteK}
              elbowK={kEvaluation.elbowK}
              selectedK={selectedK}
              businessNote={businessKNote}
              onScalingChange={(m) => {
                setScalingMethod(m);
                computeKMetricsNow(m);
                invalidateFromStep(8);
              }}
              onRecalculateK={() => computeKMetricsNow(scalingMethod)}
              onSelectK={(kVal) => {
                setSelectedK(kVal);
                invalidateFromStep(8);
              }}
              onBusinessNoteChange={setBusinessKNote}
            />
          )}

          {currentStep === 8 && (
            <Step8AlgorithmSelection
              algorithm={algorithm}
              metric={metric}
              linkage={linkage}
              nInit={nInit}
              randomState={randomState}
              maxIter={maxIter}
              tol={tol}
              onAlgorithmChange={(a) => {
                setAlgorithm(a);
                invalidateFromStep(8);
              }}
              onMetricChange={(m) => {
                setMetric(m);
                invalidateFromStep(8);
              }}
              onLinkageChange={(l) => {
                setLinkage(l);
                invalidateFromStep(8);
              }}
              onKMeansParamsChange={(field, val) => {
                if (field === 'nInit') setNInit(val);
                if (field === 'randomState') setRandomState(val);
                if (field === 'maxIter') setMaxIter(val);
                if (field === 'tol') setTol(val);
                invalidateFromStep(8);
              }}
            />
          )}

          {currentStep === 9 && (
            <Step9Execution
              k={selectedK}
              algorithm={algorithm}
              rowCount={activeModeledRows.length}
              labels={clusterLabels}
              metrics={clusterMetrics}
              labeledRows={labeledRows}
              columns={columns}
              onExecute={handleRunClustering}
            />
          )}

          {currentStep === 10 && pcaProjection && (
            <Step10CentroidsVisuals
              centroids={centroids}
              numericCols={numericCols}
              labels={clusterLabels || []}
              pca={pcaProjection}
              profiles={profiles}
            />
          )}

          {currentStep === 11 && (
            <Step11Profiling
              sector={effectiveSector}
              hasApiKey={apiKey.trim().length > 0}
              aiModeUsed={aiModeUsed}
              loading={aiLoading}
              errorMsg={aiError}
              centroids={centroids}
              profiles={profiles}
              onGenerateProfiles={handleGenerateProfiles}
            />
          )}

          {currentStep === 12 && profiles && strategies && (
            <Step12Strategies
              centroids={centroids}
              profiles={profiles}
              strategies={strategies}
            />
          )}

          {currentStep === 13 && (
            <Step13ExportReport
              projectName={projectName}
              sector={effectiveSector}
              algorithm={algorithm}
              k={selectedK}
              modeledCount={activeModeledRows.length}
              onDownloadHtml={handleDownloadHtmlReport}
            />
          )}

          {/* Navegación secuencial inferior */}
          <div className="mt-8 pt-5 border-t border-slate-200 flex items-center justify-between">
            <button
              type="button"
              disabled={currentStep <= 1}
              onClick={() => goToStep(currentStep - 1)}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed px-4 py-2.5 text-sm font-medium text-slate-700 transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
              <span>Paso anterior</span>
            </button>

            <span className="text-xs text-slate-500 font-mono tabular-nums">
              Etapa {currentStep} de 13
            </span>

            {currentStep < 13 ? (
              <button
                type="button"
                disabled={!canGoNext()}
                onClick={() => goToStep(currentStep + 1)}
                className="inline-flex items-center gap-2 rounded-lg bg-[#0033A0] hover:bg-[#002880] disabled:bg-slate-200 disabled:text-slate-500 px-5 py-2.5 text-sm font-medium text-white transition-colors"
              >
                <span>Siguiente paso</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            ) : (
              <button
                type="button"
                onClick={handleDownloadHtmlReport}
                className="inline-flex items-center gap-2 rounded-lg bg-[#0033A0] hover:bg-[#002880] px-5 py-2.5 text-sm font-medium text-white transition-colors"
              >
                <Download className="w-4 h-4" />
                <span>Exportar HTML</span>
              </button>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
