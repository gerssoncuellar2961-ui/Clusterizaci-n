export type DataRow = Record<string, string | number | null>;

export type SeparatorOption = ',' | ';' | '\t';
export type EncodingOption = 'utf-8' | 'iso-8859-1';

export type NumericImputationMethod = 'Media' | 'Mediana' | 'Cero' | 'KNNImputer';
export type CategoricalImputationMethod = 'Moda' | 'Desconocido';
export type ImputationStrategy = NumericImputationMethod | CategoricalImputationMethod;

export type ScalingMethod = 'StandardScaler' | 'MinMaxScaler';
export type ClusteringAlgorithm = 'K-Means' | 'Jerárquico / Aglomerativo';
export type DistanceMetric = 'euclidean' | 'manhattan' | 'cosine';
export type LinkageMethod = 'ward' | 'complete' | 'average' | 'single';

export interface ColumnInspection {
  variable: string;
  dataType: 'Numérica' | 'Categórica';
  nullCount: number;
  nullPct: number;
  uniqueCount: number;
  zeroVariance: boolean;
}

export interface DescriptiveStatsRow {
  variable: string;
  count: number;
  mean: number;
  std: number;
  min: number;
  q25: number;
  median: number;
  q75: number;
  max: number;
  iqrOutliers: number;
}

export interface BusinessRuleRange {
  min: number;
  max: number;
}

export interface KMetricPoint {
  k: number;
  inertia: number;
  silhouette: number;
}

export interface ClusterMetrics {
  silhouette: number;
  calinskiHarabasz: number;
  daviesBouldin: number;
  inertia: number | null;
}

export interface CentroidRow {
  clusterId: number;
  values: Record<string, number>;
  n: number;
  percentage: number;
}

export interface PCAProjection {
  points: [number, number][];
  centroidPoints: [number, number][];
  explainedVarianceRatio: [number, number];
}

export interface ClusterProfile {
  clusterId: number;
  nombreComercial: string;
  descripcionPerfil: string;
  caracteristicasDiferenciadoras: string[];
  comportamientoPredominante: string;
  necesidadesRelevantes: string;
  oportunidadesComerciales: string;
}

export interface FivePillarsStrategy {
  Producto: string[];
  Precio: string[];
  Innovación: string[];
  Comunicación: string[];
  Rentabilidad: string[];
}

export const ECONOMIC_SECTORS = [
  'Retail y comercio',
  'Banca y servicios financieros',
  'Transporte y logística',
  'Telecomunicaciones',
  'Salud',
  'Manufactura e industria',
  'Educación',
  'Seguridad y defensa',
  'Alimentos y bebidas',
  'Otro',
] as const;

export const CLUSTER_COLORS = [
  '#0033A0', // Azul Rey
  '#00B5AD', // Azul Aguamarina
  '#78BE20', // Verde Biche
  '#D97706', // Ámbar institucional
  '#6366F1', // Índigo
  '#0D9488', // Verde azulado
  '#DC2626', // Carmesí
  '#475569', // Pizarra
  '#2563EB', // Azul medio
  '#65A30D', // Verde oliva
];

export const STEP_DEFINITIONS = [
  { id: 1, short: 'Carga de archivos', title: '3.1. Carga de archivos multiformato' },
  { id: 2, short: 'Vista previa y sector', title: '3.2. Vista previa y sector económico' },
  { id: 3, short: 'Inspección de variables', title: '3.3. Inspección y clasificación de variables' },
  { id: 4, short: 'Nulos y duplicados', title: '3.4. Limpieza y diagnóstico de faltantes' },
  { id: 5, short: 'Imputación de datos', title: '3.5. Imputación de datos' },
  { id: 6, short: 'Descriptivo y reglas', title: '3.6. Análisis descriptivo y reglas de negocio' },
  { id: 7, short: 'Selección de K', title: '3.7. Determinación del número óptimo de clusters (K)' },
  { id: 8, short: 'Algoritmo y distancia', title: '3.8. Selección del algoritmo y métrica de distancia' },
  { id: 9, short: 'Ejecución del modelo', title: '3.9. Ejecución del clustering' },
  { id: 10, short: 'Centroides y gráficos', title: '3.10. Visualización de centroides y perfiles' },
  { id: 11, short: 'Perfilamiento comercial', title: '3.11. Perfilamiento automático y nombres comerciales' },
  { id: 12, short: 'Estrategias 5 pilares', title: '3.12. Generación de estrategias comerciales' },
  { id: 13, short: 'Informe HTML', title: '3.13. Exportación del informe ejecutivo en HTML' },
] as const;
