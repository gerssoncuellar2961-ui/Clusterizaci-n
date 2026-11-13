import {
  CentroidRow,
  ClusterMetrics,
  ClusteringAlgorithm,
  DataRow,
  DistanceMetric,
  KMetricPoint,
  LinkageMethod,
  PCAProjection,
  ScalingMethod,
} from '../types/clustering';

export interface ScalerModel {
  method: ScalingMethod;
  center: number[];
  scale: number[];
  transform: (matrix: number[][]) => number[][];
}

function createRng(seed: number) {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function extractNumericMatrix(rows: DataRow[], numericCols: string[]): number[][] {
  return rows.map((row) =>
    numericCols.map((col) => {
      const v = row[col];
      if (typeof v !== 'number' || !Number.isFinite(v)) {
        throw new Error(`Existen valores faltantes o no numéricos en la columna '${col}'.`);
      }
      return v;
    })
  );
}

export function fitTransformScaler(
  matrix: number[][],
  method: ScalingMethod
): { scaled: number[][]; scaler: ScalerModel } {
  const n = matrix.length;
  const d = matrix[0].length;
  const center = new Array<number>(d).fill(0);
  const scale = new Array<number>(d).fill(1);

  if (method === 'StandardScaler') {
    for (let j = 0; j < d; j++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += matrix[i][j];
      const mean = sum / n;
      let varSum = 0;
      for (let i = 0; i < n; i++) varSum += (matrix[i][j] - mean) ** 2;
      const std = Math.sqrt(varSum / n) || 1;
      center[j] = mean;
      scale[j] = std;
    }
  } else {
    for (let j = 0; j < d; j++) {
      let minV = Infinity;
      let maxV = -Infinity;
      for (let i = 0; i < n; i++) {
        if (matrix[i][j] < minV) minV = matrix[i][j];
        if (matrix[i][j] > maxV) maxV = matrix[i][j];
      }
      center[j] = minV;
      scale[j] = maxV - minV || 1;
    }
  }

  const transform = (input: number[][]) =>
    input.map((row) => row.map((val, j) => (val - center[j]) / scale[j]));

  return {
    scaled: transform(matrix),
    scaler: { method, center, scale, transform },
  };
}

function euclideanSq(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    s += diff * diff;
  }
  return s;
}

function computeDistance(a: number[], b: number[], metric: DistanceMetric): number {
  if (metric === 'euclidean') {
    return Math.sqrt(euclideanSq(a, b));
  }
  if (metric === 'manhattan') {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
    return s;
  }
  // Distancia coseno: 1 - cos(theta)
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0) return 0;
  return Math.max(0, 1 - dot / denom);
}

export function runKMeans(
  X: number[][],
  k: number,
  nInit: number,
  randomState: number,
  maxIter: number,
  tol: number
): { labels: number[]; inertia: number } {
  const n = X.length;
  const d = X[0].length;
  const rand = createRng(randomState);

  let bestLabels = new Array<number>(n).fill(0);
  let bestInertia = Infinity;

  for (let init = 0; init < nInit; init++) {
    // Inicialización K-Means++
    const centroids: number[][] = [];
    const firstIdx = Math.floor(rand() * n);
    centroids.push([...X[firstIdx]]);

    while (centroids.length < k) {
      const dists = X.map((pt) => {
        let minD = Infinity;
        for (const c of centroids) {
          const dSq = euclideanSq(pt, c);
          if (dSq < minD) minD = dSq;
        }
        return minD;
      });
      const totalDist = dists.reduce((a, b) => a + b, 0);
      if (totalDist === 0) {
        centroids.push([...X[Math.floor(rand() * n)]]);
        continue;
      }
      let target = rand() * totalDist;
      let chosen = 0;
      for (let i = 0; i < n; i++) {
        target -= dists[i];
        if (target <= 0) {
          chosen = i;
          break;
        }
      }
      centroids.push([...X[chosen]]);
    }

    const labels = new Array<number>(n).fill(0);
    for (let iter = 0; iter < maxIter; iter++) {
      // Asignación al centroide más cercano
      for (let i = 0; i < n; i++) {
        let bestC = 0;
        let bestD = Infinity;
        for (let c = 0; c < k; c++) {
          const dist = euclideanSq(X[i], centroids[c]);
          if (dist < bestD) {
            bestD = dist;
            bestC = c;
          }
        }
        labels[i] = bestC;
      }

      // Actualización de centroides
      const sums = Array.from({ length: k }, () => new Array<number>(d).fill(0));
      const counts = new Array<number>(k).fill(0);
      for (let i = 0; i < n; i++) {
        const c = labels[i];
        counts[c]++;
        for (let j = 0; j < d; j++) sums[c][j] += X[i][j];
      }

      let maxShift = 0;
      for (let c = 0; c < k; c++) {
        if (counts[c] === 0) {
          // Reubicar cluster vacío al punto más lejano para evitar colapso
          let farIdx = 0;
          let farDist = -1;
          for (let i = 0; i < n; i++) {
            const dd = euclideanSq(X[i], centroids[labels[i]]);
            if (dd > farDist) {
              farDist = dd;
              farIdx = i;
            }
          }
          centroids[c] = [...X[farIdx]];
          continue;
        }
        for (let j = 0; j < d; j++) {
          const newCoord = sums[c][j] / counts[c];
          const shift = Math.abs(newCoord - centroids[c][j]);
          if (shift > maxShift) maxShift = shift;
          centroids[c][j] = newCoord;
        }
      }

      if (maxShift <= tol) break;
    }

    let inertia = 0;
    for (let i = 0; i < n; i++) {
      inertia += euclideanSq(X[i], centroids[labels[i]]);
    }

    if (inertia < bestInertia) {
      bestInertia = inertia;
      bestLabels = [...labels];
    }
  }

  return { labels: bestLabels, inertia: bestInertia };
}

export function validateHierarchicalCombination(
  metric: DistanceMetric,
  linkage: LinkageMethod
): { valid: boolean; message: string } {
  if (linkage === 'ward' && metric !== 'euclidean') {
    return {
      valid: false,
      message:
        "El enlace 'Ward' requiere obligatoriamente distancia 'Euclidiana'. No es compatible con Manhattan ni Coseno.",
    };
  }
  return { valid: true, message: 'Combinación válida entre métrica y enlace.' };
}

export function runAgglomerative(
  X: number[][],
  k: number,
  metric: DistanceMetric,
  linkage: LinkageMethod
): number[] {
  const check = validateHierarchicalCombination(metric, linkage);
  if (!check.valid) {
    throw new Error(check.message);
  }

  const n = X.length;
  // Cada punto inicia como su propio cluster
  const clusters: { members: number[]; centroid: number[] }[] = X.map((pt, idx) => ({
    members: [idx],
    centroid: [...pt],
  }));

  // Matriz de distancias iniciales entre puntos
  const pairDist: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = computeDistance(X[i], X[j], metric);
      pairDist[i][j] = d;
      pairDist[j][i] = d;
    }
  }

  const clusterDistance = (
    c1: { members: number[]; centroid: number[] },
    c2: { members: number[]; centroid: number[] }
  ): number => {
    if (linkage === 'ward') {
      const n1 = c1.members.length;
      const n2 = c2.members.length;
      const dSq = euclideanSq(c1.centroid, c2.centroid);
      return Math.sqrt((2 * n1 * n2) / (n1 + n2) * dSq);
    }
    if (linkage === 'single') {
      let minD = Infinity;
      for (const a of c1.members) {
        for (const b of c2.members) {
          if (pairDist[a][b] < minD) minD = pairDist[a][b];
        }
      }
      return minD;
    }
    if (linkage === 'complete') {
      let maxD = -Infinity;
      for (const a of c1.members) {
        for (const b of c2.members) {
          if (pairDist[a][b] > maxD) maxD = pairDist[a][b];
        }
      }
      return maxD;
    }
    // average
    let sum = 0;
    for (const a of c1.members) {
      for (const b of c2.members) {
        sum += pairDist[a][b];
      }
    }
    return sum / (c1.members.length * c2.members.length);
  };

  while (clusters.length > k) {
    let bestI = 0;
    let bestJ = 1;
    let bestDist = Infinity;

    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const dist = clusterDistance(clusters[i], clusters[j]);
        if (dist < bestDist) {
          bestDist = dist;
          bestI = i;
          bestJ = j;
        }
      }
    }

    const mergedMembers = [...clusters[bestI].members, ...clusters[bestJ].members];
    const dim = X[0].length;
    const newCentroid = new Array<number>(dim).fill(0);
    for (const idx of mergedMembers) {
      for (let d = 0; d < dim; d++) newCentroid[d] += X[idx][d];
    }
    for (let d = 0; d < dim; d++) newCentroid[d] /= mergedMembers.length;

    clusters[bestI] = { members: mergedMembers, centroid: newCentroid };
    clusters.splice(bestJ, 1);
  }

  const labels = new Array<number>(n).fill(0);
  clusters.forEach((c, cid) => {
    for (const memberIdx of c.members) {
      labels[memberIdx] = cid;
    }
  });
  return labels;
}

export function computeSilhouetteScore(
  X: number[][],
  labels: number[],
  metric: DistanceMetric = 'euclidean'
): number {
  const n = X.length;
  const groups = new Map<number, number[]>();
  labels.forEach((lbl, idx) => {
    if (!groups.has(lbl)) groups.set(lbl, []);
    groups.get(lbl)!.push(idx);
  });
  if (groups.size < 2 || groups.size >= n) return 0;

  let totalScore = 0;
  for (let i = 0; i < n; i++) {
    const ownLabel = labels[i];
    const ownGroup = groups.get(ownLabel)!;
    if (ownGroup.length <= 1) continue;

    let a = 0;
    for (const j of ownGroup) {
      if (i !== j) a += computeDistance(X[i], X[j], metric);
    }
    a /= ownGroup.length - 1;

    let b = Infinity;
    for (const [lbl, members] of groups.entries()) {
      if (lbl === ownLabel) continue;
      let sum = 0;
      for (const j of members) sum += computeDistance(X[i], X[j], metric);
      const meanDist = sum / members.length;
      if (meanDist < b) b = meanDist;
    }

    const denom = Math.max(a, b);
    totalScore += denom > 0 ? (b - a) / denom : 0;
  }

  return totalScore / n;
}

export function computeClusterEvaluationMetrics(
  X: number[][],
  labels: number[],
  metric: DistanceMetric,
  inertia: number | null
): ClusterMetrics {
  const n = X.length;
  const d = X[0].length;
  const silhouette = computeSilhouetteScore(X, labels, metric);

  const uniqueLabels = Array.from(new Set(labels)).sort((a, b) => a - b);
  const k = uniqueLabels.length;

  const globalMean = new Array<number>(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) globalMean[j] += X[i][j];
  }
  for (let j = 0; j < d; j++) globalMean[j] /= n;

  const centroids = new Map<number, number[]>();
  const counts = new Map<number, number>();
  for (const lbl of uniqueLabels) {
    centroids.set(lbl, new Array<number>(d).fill(0));
    counts.set(lbl, 0);
  }
  for (let i = 0; i < n; i++) {
    const lbl = labels[i];
    counts.set(lbl, counts.get(lbl)! + 1);
    const c = centroids.get(lbl)!;
    for (let j = 0; j < d; j++) c[j] += X[i][j];
  }
  for (const lbl of uniqueLabels) {
    const cnt = counts.get(lbl)!;
    const c = centroids.get(lbl)!;
    for (let j = 0; j < d; j++) c[j] /= cnt;
  }

  let ssb = 0;
  let ssw = 0;
  for (const lbl of uniqueLabels) {
    ssb += counts.get(lbl)! * euclideanSq(centroids.get(lbl)!, globalMean);
  }
  for (let i = 0; i < n; i++) {
    ssw += euclideanSq(X[i], centroids.get(labels[i])!);
  }

  const calinskiHarabasz = k > 1 && n > k && ssw > 0 ? (ssb / (k - 1)) / (ssw / (n - k)) : 0;

  // Davies-Bouldin
  const scatter = new Map<number, number>();
  for (const lbl of uniqueLabels) {
    const c = centroids.get(lbl)!;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      if (labels[i] === lbl) sum += Math.sqrt(euclideanSq(X[i], c));
    }
    scatter.set(lbl, sum / counts.get(lbl)!);
  }

  let dbSum = 0;
  for (const ci of uniqueLabels) {
    let maxRatio = 0;
    for (const cj of uniqueLabels) {
      if (ci === cj) continue;
      const dist = Math.sqrt(euclideanSq(centroids.get(ci)!, centroids.get(cj)!));
      if (dist > 0) {
        const ratio = (scatter.get(ci)! + scatter.get(cj)!) / dist;
        if (ratio > maxRatio) maxRatio = ratio;
      }
    }
    dbSum += maxRatio;
  }
  const daviesBouldin = k > 0 ? dbSum / k : 0;

  return {
    silhouette: Number(silhouette.toFixed(4)),
    calinskiHarabasz: Number(calinskiHarabasz.toFixed(2)),
    daviesBouldin: Number(daviesBouldin.toFixed(4)),
    inertia: inertia !== null ? Number(inertia.toFixed(2)) : null,
  };
}

export function evaluateKRange(
  X: number[][],
  kMax = 10
): { points: KMetricPoint[]; bestSilhouetteK: number; elbowK: number } {
  const maxAllowed = Math.min(kMax, X.length - 1);
  const points: KMetricPoint[] = [];

  for (let k = 2; k <= maxAllowed; k++) {
    const { labels, inertia } = runKMeans(X, k, 6, 42, 150, 1e-4);
    const silhouette = computeSilhouetteScore(X, labels, 'euclidean');
    points.push({
      k,
      inertia: Number(inertia.toFixed(2)),
      silhouette: Number(silhouette.toFixed(4)),
    });
  }

  let bestSilhouetteK = 2;
  let maxSil = -Infinity;
  for (const p of points) {
    if (p.silhouette > maxSil) {
      maxSil = p.silhouette;
      bestSilhouetteK = p.k;
    }
  }

  // Detección de codo mediante distancia perpendicular máxima a la recta entre los extremos
  let elbowK = bestSilhouetteK;
  if (points.length >= 3) {
    const p1 = points[0];
    const p2 = points[points.length - 1];
    const dx = p2.k - p1.k;
    const dy = p2.inertia - p1.inertia;
    const norm = Math.sqrt(dx * dx + dy * dy);
    if (norm > 0) {
      let maxDist = -1;
      for (const p of points) {
        const dist = Math.abs(dy * p.k - dx * p.inertia + p2.k * p1.inertia - p2.inertia * p1.k) / norm;
        if (dist > maxDist) {
          maxDist = dist;
          elbowK = p.k;
        }
      }
    }
  }

  return { points, bestSilhouetteK, elbowK };
}

export function computeUnscaledCentroids(
  rows: DataRow[],
  labels: number[],
  numericCols: string[]
): CentroidRow[] {
  const uniqueClusters = Array.from(new Set(labels)).sort((a, b) => a - b);
  const total = rows.length;

  return uniqueClusters.map((cid) => {
    const sums: Record<string, number> = {};
    for (const col of numericCols) sums[col] = 0;
    let count = 0;

    for (let i = 0; i < rows.length; i++) {
      if (labels[i] === cid) {
        count++;
        for (const col of numericCols) {
          sums[col] += Number(rows[i][col]);
        }
      }
    }

    const values: Record<string, number> = {};
    for (const col of numericCols) {
      values[col] = Number((sums[col] / Math.max(1, count)).toFixed(2));
    }

    return {
      clusterId: cid,
      values,
      n: count,
      percentage: Number(((count / Math.max(1, total)) * 100).toFixed(2)),
    };
  });
}

export function computePCA2D(
  X: number[][],
  centroidsUnscaled: CentroidRow[],
  numericCols: string[],
  scaler: ScalerModel
): PCAProjection {
  const n = X.length;
  const d = X[0].length;

  // Centrar los datos escalados
  const means = new Array<number>(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) means[j] += X[i][j];
  }
  for (let j = 0; j < d; j++) means[j] /= n;

  const Xc = X.map((row) => row.map((v, j) => v - means[j]));

  // Matriz de covarianza d x d
  const cov: number[][] = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  for (let a = 0; a < d; a++) {
    for (let b = a; b < d; b++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += Xc[i][a] * Xc[i][b];
      const val = s / Math.max(1, n - 1);
      cov[a][b] = val;
      cov[b][a] = val;
    }
  }

  // Algoritmo de Jacobi para autovalores y autovectores reales simétricos
  const V: number[][] = Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (_, j) => (i === j ? 1 : 0))
  );
  const A = cov.map((r) => [...r]);

  for (let iter = 0; iter < 80; iter++) {
    let p = 0;
    let q = 1;
    let maxOff = 0;
    for (let i = 0; i < d; i++) {
      for (let j = i + 1; j < d; j++) {
        if (Math.abs(A[i][j]) > maxOff) {
          maxOff = Math.abs(A[i][j]);
          p = i;
          q = j;
        }
      }
    }
    if (maxOff < 1e-10) break;

    const app = A[p][p];
    const aqq = A[q][q];
    const apq = A[p][q];
    const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(phi);
    const s = Math.sin(phi);

    for (let i = 0; i < d; i++) {
      if (i !== p && i !== q) {
        const aip = A[i][p];
        const aiq = A[i][q];
        A[i][p] = c * aip - s * aiq;
        A[p][i] = A[i][p];
        A[i][q] = s * aip + c * aiq;
        A[q][i] = A[i][q];
      }
    }
    A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    A[p][q] = 0;
    A[q][p] = 0;

    for (let i = 0; i < d; i++) {
      const vip = V[i][p];
      const viq = V[i][q];
      V[i][p] = c * vip - s * viq;
      V[i][q] = s * vip + c * viq;
    }
  }

  const eigenPairs = Array.from({ length: d }, (_, j) => ({
    val: Math.max(0, A[j][j]),
    vec: V.map((row) => row[j]),
  })).sort((a, b) => b.val - a.val);

  const totalVar = eigenPairs.reduce((acc, e) => acc + e.val, 0) || 1;
  const pc1 = eigenPairs[0].vec;
  const pc2 = eigenPairs[1]?.vec || new Array<number>(d).fill(0);

  const projectVec = (vec: number[]): [number, number] => {
    let x1 = 0;
    let x2 = 0;
    for (let j = 0; j < d; j++) {
      const centered = vec[j] - means[j];
      x1 += centered * pc1[j];
      x2 += centered * pc2[j];
    }
    return [Number(x1.toFixed(4)), Number(x2.toFixed(4))];
  };

  const points = X.map((row) => projectVec(row));

  const centroidsRawMatrix = centroidsUnscaled.map((c) =>
    numericCols.map((col) => c.values[col])
  );
  const centroidsScaled = scaler.transform(centroidsRawMatrix);
  const centroidPoints = centroidsScaled.map((row) => projectVec(row));

  return {
    points,
    centroidPoints,
    explainedVarianceRatio: [
      eigenPairs[0].val / totalVar,
      (eigenPairs[1]?.val || 0) / totalVar,
    ],
  };
}

export function executeClusteringPipeline(params: {
  rows: DataRow[];
  numericCols: string[];
  scalingMethod: ScalingMethod;
  algorithm: ClusteringAlgorithm;
  k: number;
  metric: DistanceMetric;
  linkage: LinkageMethod;
  nInit: number;
  randomState: number;
  maxIter: number;
  tol: number;
}) {
  const {
    rows,
    numericCols,
    scalingMethod,
    algorithm,
    k,
    metric,
    linkage,
    nInit,
    randomState,
    maxIter,
    tol,
  } = params;

  const rawMatrix = extractNumericMatrix(rows, numericCols);
  const { scaled, scaler } = fitTransformScaler(rawMatrix, scalingMethod);

  let labels: number[];
  let inertia: number | null = null;

  if (algorithm === 'K-Means') {
    const res = runKMeans(scaled, k, nInit, randomState, maxIter, tol);
    labels = res.labels;
    inertia = res.inertia;
  } else {
    labels = runAgglomerative(scaled, k, metric, linkage);
  }

  const metrics = computeClusterEvaluationMetrics(scaled, labels, metric, inertia);
  const centroids = computeUnscaledCentroids(rows, labels, numericCols);
  const pca = computePCA2D(scaled, centroids, numericCols, scaler);

  return { labels, metrics, centroids, pca };
}
