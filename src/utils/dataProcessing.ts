import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import {
  BusinessRuleRange,
  ColumnInspection,
  DataRow,
  DescriptiveStatsRow,
  EncodingOption,
  ImputationStrategy,
  SeparatorOption,
} from '../types/clustering';

function normalizeCell(val: unknown): string | number | null {
  if (val === null || val === undefined) return null;
  if (typeof val === 'number') {
    return Number.isFinite(val) ? val : null;
  }
  const str = String(val).trim();
  if (str === '' || str.toLowerCase() === 'nan' || str.toLowerCase() === 'null' || str.toLowerCase() === 'na') {
    return null;
  }
  // Detectar números con formato estándar sin confundir IDs alfanuméricos
  if (/^-?\d+(\.\d+)?$/.test(str)) {
    const num = Number(str);
    return Number.isFinite(num) ? num : str;
  }
  return str;
}

export async function parseUploadedFile(
  file: File,
  separator: SeparatorOption,
  encoding: EncodingOption
): Promise<{ rows: DataRow[]; columns: string[] }> {
  const lower = file.name.toLowerCase();
  const buffer = await file.arrayBuffer();
  if (!buffer || buffer.byteLength === 0) {
    throw new Error('El archivo cargado está completamente vacío.');
  }

  let rawObjects: Record<string, unknown>[] = [];

  if (lower.endsWith('.csv') || lower.endsWith('.txt')) {
    const decoder = new TextDecoder(encoding);
    const text = decoder.decode(buffer);
    if (!text.trim()) {
      throw new Error('El archivo de texto no contiene registros.');
    }
    const parsed = Papa.parse<Record<string, unknown>>(text, {
      header: true,
      delimiter: separator,
      skipEmptyLines: true,
    });
    if (parsed.errors.length > 0 && parsed.data.length === 0) {
      throw new Error(`No se pudo analizar el archivo: ${parsed.errors[0].message}`);
    }
    rawObjects = parsed.data;
  } else if (lower.endsWith('.xlsx') || lower.endsWith('.xls')) {
    const workbook = XLSX.read(buffer, { type: 'array' });
    if (!workbook.SheetNames.length) {
      throw new Error('El libro de Excel no contiene hojas válidas.');
    }
    const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
    rawObjects = XLSX.utils.sheet_to_json<Record<string, unknown>>(firstSheet, { defval: null });
  } else {
    throw new Error('Formato no compatible. Seleccione un archivo CSV, TXT, XLS o XLSX.');
  }

  if (!rawObjects.length) {
    throw new Error('El archivo no contiene filas de datos válidas.');
  }

  const columns = Object.keys(rawObjects[0]).filter((c) => c.trim() !== '');
  if (!columns.length) {
    throw new Error('No se detectaron columnas con encabezados válidos.');
  }

  const rows: DataRow[] = rawObjects.map((obj) => {
    const row: DataRow = {};
    for (const col of columns) {
      row[col] = normalizeCell(obj[col]);
    }
    return row;
  });

  return { rows, columns };
}

export function generateSampleDataset(): { rows: DataRow[]; columns: string[] } {
  // Generador pseudoaleatorio determinístico (Mulberry32) para pruebas reproducibles
  let seed = 42;
  const rand = () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const randn = (mean: number, std: number) => {
    const u1 = Math.max(1e-7, rand());
    const u2 = rand();
    const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + z * std;
  };

  const canales = ['Tienda Física', 'E-Commerce', 'App Móvil', 'Mixto'];
  const ciudades = ['Bogotá', 'Medellín', 'Cali', 'Barranquilla'];
  const rows: DataRow[] = [];

  for (let i = 0; i < 180; i++) {
    const r = rand();
    const seg = r < 0.3 ? 0 : r < 0.6 ? 1 : r < 0.85 ? 2 : 3;

    const edad = Math.max(18, Math.round(
      seg === 0 ? randn(28, 5) : seg === 1 ? randn(42, 6) : seg === 2 ? randn(54, 7) : randn(35, 8)
    ));
    const ingreso = Number(Math.max(900,
      seg === 0 ? randn(2800, 500) : seg === 1 ? randn(6200, 900) : seg === 2 ? randn(9500, 1200) : randn(4100, 700)
    ).toFixed(2));
    const frecuencia = Math.max(1, Math.min(40, Math.round(
      seg === 0 ? randn(4, 1.5) : seg === 1 ? randn(12, 2.5) : seg === 2 ? randn(22, 4) : randn(2, 1)
    )));
    const ticket = Number(Math.max(15, Math.min(900,
      seg === 0 ? randn(65, 18) : seg === 1 ? randn(185, 35) : seg === 2 ? randn(410, 70) : randn(95, 25)
    )).toFixed(2));
    const antiguedad = Math.max(1, Math.min(120, Math.round(
      seg === 0 ? randn(10, 4) : seg === 1 ? randn(32, 8) : seg === 2 ? randn(64, 12) : randn(6, 3)
    )));

    rows.push({
      ID_Cliente: `CLI-${1000 + i}`,
      Edad: edad,
      Ingreso_Mensual_USD: ingreso,
      Frecuencia_Compra_Anual: frecuencia,
      Ticket_Promedio_USD: ticket,
      Antiguedad_Meses: antiguedad,
      Canal_Preferido: canales[Math.floor(rand() * canales.length)],
      Ciudad: ciudades[Math.floor(rand() * ciudades.length)],
    });
  }

  // Nulos, valores atípicos de negocio y duplicados de prueba
  rows[3].Ingreso_Mensual_USD = null;
  rows[18].Ingreso_Mensual_USD = null;
  rows[27].Ticket_Promedio_USD = null;
  rows[45].Canal_Preferido = null;
  rows[10].Edad = 145;
  rows[75].Ticket_Promedio_USD = -25;
  rows.push({ ...rows[0] }, { ...rows[5] }, { ...rows[12] });

  return {
    rows,
    columns: Object.keys(rows[0]),
  };
}

export function inspectColumns(rows: DataRow[], columns: string[]): ColumnInspection[] {
  const total = Math.max(rows.length, 1);
  return columns.map((col) => {
    let nullCount = 0;
    let numericCount = 0;
    let nonNullCount = 0;
    const uniqueSet = new Set<string | number>();

    for (const row of rows) {
      const val = row[col];
      if (val === null || val === undefined || val === '') {
        nullCount++;
      } else {
        nonNullCount++;
        uniqueSet.add(val);
        if (typeof val === 'number' && Number.isFinite(val)) {
          numericCount++;
        }
      }
    }

    const isNumeric = nonNullCount > 0 && numericCount === nonNullCount;
    return {
      variable: col,
      dataType: isNumeric ? 'Numérica' : 'Categórica',
      nullCount,
      nullPct: Number(((nullCount / total) * 100).toFixed(2)),
      uniqueCount: uniqueSet.size,
      zeroVariance: uniqueSet.size <= 1,
    };
  });
}

export function countDuplicates(rows: DataRow[], columns: string[]): number {
  const seen = new Set<string>();
  let duplicates = 0;
  for (const row of rows) {
    const key = JSON.stringify(columns.map((c) => row[c]));
    if (seen.has(key)) {
      duplicates++;
    } else {
      seen.add(key);
    }
  }
  return duplicates;
}

export function removeDuplicateRows(rows: DataRow[], columns: string[]): { cleaned: DataRow[]; removed: number } {
  const seen = new Set<string>();
  const cleaned: DataRow[] = [];
  let removed = 0;
  for (const row of rows) {
    const key = JSON.stringify(columns.map((c) => row[c]));
    if (seen.has(key)) {
      removed++;
    } else {
      seen.add(key);
      cleaned.push({ ...row });
    }
  }
  return { cleaned, removed };
}

function getMedian(sorted: number[]): number {
  if (!sorted.length) return 0;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function applyImputation(
  rows: DataRow[],
  columns: string[],
  inspection: ColumnInspection[],
  strategies: Record<string, ImputationStrategy>,
  dropRowsWithNulls: boolean
): { imputed: DataRow[]; summary: string[] } {
  const summary: string[] = [];

  if (dropRowsWithNulls) {
    const filtered = rows
      .filter((r) => columns.every((c) => r[c] !== null && r[c] !== undefined && r[c] !== ''))
      .map((r) => ({ ...r }));
    const removed = rows.length - filtered.length;
    summary.push(`Se eliminaron ${removed} registros con valores faltantes (quedan ${filtered.length} registros).`);
    return { imputed: filtered, summary };
  }

  const working = rows.map((r) => ({ ...r }));
  const numericCols = inspection.filter((c) => c.dataType === 'Numérica').map((c) => c.variable);
  const colsWithNulls = inspection.filter((c) => c.nullCount > 0);

  if (!colsWithNulls.length) {
    return { imputed: working, summary: ['No se detectaron valores faltantes en la base.'] };
  }

  // Pre-calcular medias para KNNImputer
  const knnCols = colsWithNulls
    .filter((c) => strategies[c.variable] === 'KNNImputer')
    .map((c) => c.variable);

  if (knnCols.length > 0 && numericCols.length > 0) {
    // Normalizar columnas numéricas para distancia euclidiana equilibrada en KNNImputer
    const colMeans: Record<string, number> = {};
    const colStds: Record<string, number> = {};
    for (const col of numericCols) {
      const vals = working.map((r) => r[col]).filter((v): v is number => typeof v === 'number');
      const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
      const variance = vals.length ? vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length : 1;
      colMeans[col] = mean;
      colStds[col] = Math.sqrt(variance) || 1;
    }

    const originalCopy = working.map((r) => ({ ...r }));

    for (const targetCol of knnCols) {
      let imputedCount = 0;
      for (let i = 0; i < originalCopy.length; i++) {
        if (originalCopy[i][targetCol] === null || originalCopy[i][targetCol] === undefined) {
          const neighbors: { dist: number; val: number }[] = [];
          for (let j = 0; j < originalCopy.length; j++) {
            if (i === j) continue;
            const candidateVal = originalCopy[j][targetCol];
            if (typeof candidateVal !== 'number') continue;

            let sumSq = 0;
            let shared = 0;
            for (const c of numericCols) {
              if (c === targetCol) continue;
              const vi = originalCopy[i][c];
              const vj = originalCopy[j][c];
              if (typeof vi === 'number' && typeof vj === 'number') {
                const diff = (vi - vj) / colStds[c];
                sumSq += diff * diff;
                shared++;
              }
            }
            const dist = shared > 0 ? Math.sqrt(sumSq / shared) : 0;
            neighbors.push({ dist, val: candidateVal });
          }

          neighbors.sort((a, b) => a.dist - b.dist);
          const topK = neighbors.slice(0, Math.min(5, neighbors.length));
          const imputedVal = topK.length
            ? topK.reduce((acc, item) => acc + item.val, 0) / topK.length
            : colMeans[targetCol];
          working[i][targetCol] = Number(imputedVal.toFixed(4));
          imputedCount++;
        }
      }
      summary.push(`${targetCol}: ${imputedCount} nulos imputados mediante KNNImputer (k=5 vecinos más cercanos).`);
    }
  }

  for (const info of colsWithNulls) {
    const col = info.variable;
    const strat = strategies[col] || (info.dataType === 'Numérica' ? 'Mediana' : 'Moda');
    if (strat === 'KNNImputer') continue;

    if (info.dataType === 'Numérica') {
      const vals = working.map((r) => r[col]).filter((v): v is number => typeof v === 'number');
      const sorted = [...vals].sort((a, b) => a - b);
      let fillVal = 0;
      if (strat === 'Media') {
        fillVal = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
      } else if (strat === 'Mediana') {
        fillVal = getMedian(sorted);
      } else {
        fillVal = 0;
      }
      fillVal = Number(fillVal.toFixed(4));
      let count = 0;
      for (const row of working) {
        if (row[col] === null || row[col] === undefined) {
          row[col] = fillVal;
          count++;
        }
      }
      summary.push(`${col}: ${count} nulos imputados con ${strat} (${fillVal.toLocaleString('es-CO')}).`);
    } else {
      let fillStr = 'Desconocido';
      if (strat === 'Moda') {
        const freq = new Map<string, number>();
        for (const row of working) {
          const val = row[col];
          if (val !== null && val !== undefined && val !== '') {
            const s = String(val);
            freq.set(s, (freq.get(s) || 0) + 1);
          }
        }
        let best = 'Desconocido';
        let bestCount = -1;
        for (const [k, cnt] of freq.entries()) {
          if (cnt > bestCount) {
            bestCount = cnt;
            best = k;
          }
        }
        fillStr = best;
      }
      let count = 0;
      for (const row of working) {
        if (row[col] === null || row[col] === undefined || row[col] === '') {
          row[col] = fillStr;
          count++;
        }
      }
      summary.push(`${col}: ${count} nulos imputados con ${strat} ('${fillStr}').`);
    }
  }

  return { imputed: working, summary };
}

export function computeDescriptiveStats(rows: DataRow[], numericCols: string[]): DescriptiveStatsRow[] {
  return numericCols.map((col) => {
    const vals = rows
      .map((r) => r[col])
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
      .sort((a, b) => a - b);

    const n = vals.length;
    if (n === 0) {
      return {
        variable: col,
        count: 0,
        mean: 0,
        std: 0,
        min: 0,
        q25: 0,
        median: 0,
        q75: 0,
        max: 0,
        iqrOutliers: 0,
      };
    }

    const mean = vals.reduce((a, b) => a + b, 0) / n;
    const variance = n > 1 ? vals.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1) : 0;
    const std = Math.sqrt(variance);

    const quantile = (q: number) => {
      const pos = (n - 1) * q;
      const base = Math.floor(pos);
      const rest = pos - base;
      if (vals[base + 1] !== undefined) {
        return vals[base] + rest * (vals[base + 1] - vals[base]);
      }
      return vals[base];
    };

    const q25 = quantile(0.25);
    const median = quantile(0.5);
    const q75 = quantile(0.75);
    const iqr = q75 - q25;
    const lowFence = q25 - 1.5 * iqr;
    const highFence = q75 + 1.5 * iqr;
    const iqrOutliers = vals.filter((v) => v < lowFence || v > highFence).length;

    return {
      variable: col,
      count: n,
      mean,
      std,
      min: vals[0],
      q25,
      median,
      q75,
      max: vals[n - 1],
      iqrOutliers,
    };
  });
}

export function applyBusinessRules(
  rows: DataRow[],
  rules: Record<string, BusinessRuleRange>
): { filtered: DataRow[]; ruleSummary: string[] } {
  const excludedPerCol: Record<string, number> = {};
  for (const col of Object.keys(rules)) {
    excludedPerCol[col] = 0;
  }

  const filtered = rows.filter((row) => {
    let keep = true;
    for (const [col, range] of Object.entries(rules)) {
      const val = row[col];
      if (typeof val !== 'number' || val < range.min || val > range.max) {
        excludedPerCol[col]++;
        keep = false;
      }
    }
    return keep;
  }).map((r) => ({ ...r }));

  const ruleSummary: string[] = [];
  for (const [col, count] of Object.entries(excludedPerCol)) {
    if (count > 0) {
      const r = rules[col];
      ruleSummary.push(`${col} fuera de [${r.min.toLocaleString('es-CO')}, ${r.max.toLocaleString('es-CO')}]: ${count} registros excluidos.`);
    }
  }

  if (ruleSummary.length === 0) {
    ruleSummary.push('Todos los registros cumplen con los rangos operativos configurados.');
  }

  return { filtered, ruleSummary };
}
