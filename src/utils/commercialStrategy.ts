import {
  CLUSTER_COLORS,
  CentroidRow,
  ClusterMetrics,
  ClusterProfile,
  ClusteringAlgorithm,
  DataRow,
  DistanceMetric,
  FivePillarsStrategy,
  LinkageMethod,
  PCAProjection,
  ScalingMethod,
} from '../types/clustering';

function formatNum(val: number): string {
  return val.toLocaleString('es-CO', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function generateSimulationProfilesAndStrategies(
  centroids: CentroidRow[],
  numericCols: string[],
  rows: DataRow[],
  sector: string
): {
  profiles: Record<number, ClusterProfile>;
  strategies: Record<number, FivePillarsStrategy>;
} {
  // Calcular medias y desviaciones globales para contextualizar cada centroide
  const globalMeans: Record<string, number> = {};
  const globalStds: Record<string, number> = {};

  for (const col of numericCols) {
    const vals = rows.map((r) => Number(r[col])).filter((v) => Number.isFinite(v));
    const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    const variance = vals.length
      ? vals.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, vals.length)
      : 1;
    globalMeans[col] = mean;
    globalStds[col] = Math.sqrt(variance) || 1;
  }

  const baseArchetypes = [
    'Consolidado de Alto Desempeño',
    'Expansivo en Desarrollo',
    'Operación Estandarizada',
    'Nicho Especializado',
    'Sensible a Eficiencia',
    'Tracción Moderada',
    'Emergente Focalizado',
    'Base Tradicional',
    'Alto Volumen Selectivo',
    'Segmento Conservador',
  ];

  const profiles: Record<number, ClusterProfile> = {};
  const strategies: Record<number, FivePillarsStrategy> = {};

  for (const c of centroids) {
    const zEntries = numericCols.map((col) => ({
      col,
      val: c.values[col],
      z: (c.values[col] - globalMeans[col]) / globalStds[col],
    }));

    zEntries.sort((a, b) => b.z - a.z);
    const topVar = zEntries[0];
    const bottomVar = zEntries[zEntries.length - 1];
    const avgZ = zEntries.reduce((acc, item) => acc + item.z, 0) / zEntries.length;

    let archetype = baseArchetypes[c.clusterId % baseArchetypes.length];
    if (avgZ > 0.45) {
      archetype = 'Líder de Alto Valor';
    } else if (avgZ < -0.45) {
      archetype = 'Conservador / Contención';
    } else if (topVar.z > 0.7) {
      archetype = `Especializado en ${topVar.col}`;
    }

    const nombreComercial = `${archetype} · C${c.clusterId}`;
    const altas = zEntries
      .filter((e) => e.z >= 0.35)
      .map((e) => `${e.col} (${formatNum(e.val)}, +${e.z.toFixed(1)}σ)`);
    const bajas = zEntries
      .filter((e) => e.z <= -0.35)
      .map((e) => `${e.col} (${formatNum(e.val)}, ${e.z.toFixed(1)}σ)`);

    const descAltas = altas.length
      ? altas.join(', ')
      : `niveles cercanos al promedio en ${topVar.col}`;
    const descBajas = bajas.length
      ? bajas.join(', ')
      : `sin rezagos pronunciados (menor nivel relativo en ${bottomVar.col})`;

    profiles[c.clusterId] = {
      clusterId: c.clusterId,
      nombreComercial,
      descripcionPerfil: `Segmento compuesto por ${c.n} registros (${c.percentage.toFixed(1)}% del total analizado) dentro del sector ${sector}. Se distingue estadísticamente por valores superiores en ${descAltas} y registros más contenidos en ${descBajas}.`,
      caracteristicasDiferenciadoras: [
        `Variable sobresaliente: ${topVar.col} con media de ${formatNum(topVar.val)} (desviación estándar z = ${topVar.z >= 0 ? '+' : ''}${topVar.z.toFixed(2)}).`,
        `Variable de menor magnitud relativa: ${bottomVar.col} con media de ${formatNum(bottomVar.val)} (z = ${bottomVar.z >= 0 ? '+' : ''}${bottomVar.z.toFixed(2)}).`,
        `Participación sobre la base modelada: ${c.percentage.toFixed(1)}% (${c.n} registros).`,
      ],
      comportamientoPredominante: `Comportamiento concentrado alrededor de ${topVar.col} (${formatNum(topVar.val)}), manteniendo una dinámica operativa consistente con las condiciones de ${sector}.`,
      necesidadesRelevantes: `Requiere soluciones ajustadas a su escala actual en ${topVar.col} y acompañamiento progresivo en ${bottomVar.col} sin sobredimensionar la complejidad operativa.`,
      oportunidadesComerciales: `Capitalizar la tracción observada en ${topVar.col} mediante ofertas diferenciadas y diseñar incentivos graduales para dinamizar ${bottomVar.col}.`,
    };

    strategies[c.clusterId] = {
      Producto: [
        `Priorizar soluciones centrales del portafolio de ${sector} alineadas con el desempeño observado en ${topVar.col} (${formatNum(topVar.val)}).`,
        `Estructurar paquetes escalonados que integren la oferta principal con módulos de entrada enfocados en fortalecer ${bottomVar.col}.`,
        `Habilitar esquemas de venta cruzada compatibles con los atributos predominantes de este segmento.`,
      ],
      Precio: [
        `Alinear la propuesta tarifaria con la escala observada en ${topVar.col} (${formatNum(topVar.val)}), evitando ajustes uniformes para toda la base.`,
        avgZ > 0.3
          ? 'Evitar descuentos indiscriminados dada la tracción superior observada; privilegiar beneficios por permanencia o volumen.'
          : `Diseñar planes modulares o promociones acotadas que reduzcan barreras de entrada asociadas a ${bottomVar.col}.`,
        `Evaluar el impacto de cambios de precio mediante pruebas piloto antes de extenderlos a todo el grupo (${c.n} registros).`,
      ],
      Innovación: [
        `Automatizar flujos de atención y seguimiento comercial específicos para este segmento (${c.percentage.toFixed(1)}% de la base).`,
        `Implementar alertas tempranas ante variaciones relevantes en ${topVar.col} y ${bottomVar.col}.`,
        `Desarrollar canales digitales o funcionalidades de autogestión adaptadas a la realidad operativa de ${sector}.`,
      ],
      Comunicación: [
        `Articular mensajes comerciales centrados en resultados tangibles vinculados con ${topVar.col}.`,
        `Seleccionar canales directos y digitales de contacto proporcionales al tamaño del segmento (${c.n} registros).`,
        `Ejecutar campañas de fidelización y reactivación diferenciadas respecto a los demás clusters identificados.`,
      ],
      Rentabilidad: [
        `Asignar esfuerzos y presupuesto comercial en proporción a la participación del segmento (${c.percentage.toFixed(1)}%).`,
        `Optimizar costos de atención estandarizando procesos en las variables de menor variabilidad.`,
        `Validar periódicamente la evolución de ${topVar.col} y ${bottomVar.col} con métricas reales antes de comprometer recursos adicionales.`,
      ],
    };
  }

  return { profiles, strategies };
}

export async function generateOpenAIProfilesAndStrategies(
  apiKey: string,
  model: string,
  centroids: CentroidRow[],
  numericCols: string[],
  sector: string
): Promise<{
  profiles: Record<number, ClusterProfile>;
  strategies: Record<number, FivePillarsStrategy>;
}> {
  const tableData = centroids.map((c) => ({
    Cluster: c.clusterId,
    ...c.values,
    N: c.n,
    Porcentaje: c.percentage,
  }));

  const prompt = `Actúa como un Director de Estrategia Comercial y Científico de Datos Senior.
Debes interpretar los resultados de un modelo de clustering para el sector económico: "${sector}".

Número total de clusters: ${centroids.length}
Variables numéricas utilizadas: ${numericCols.join(', ')}

Tabla de centroides desescalados (medias reales en unidades originales) con N y Porcentaje (%):
${JSON.stringify(tableData, null, 2)}

Reglas analíticas estrictas:
1. Diferencia claramente entre hallazgos estadísticos observados y recomendaciones comerciales.
2. No inventes cifras financieras ni relaciones causales que no puedan sustentarse en los datos.
3. Contextualiza las recomendaciones al sector "${sector}".
4. Diferencia claramente cada cluster según sus centroides.
5. Responde ÚNICAMENTE con un objeto JSON válido con esta estructura exacta:
{
  "clusters": [
    {
      "cluster_id": 0,
      "nombre_comercial": "...",
      "descripcion_perfil": "...",
      "caracteristicas_diferenciadoras": ["...", "...", "..."],
      "comportamiento_predominante": "...",
      "necesidades_relevantes": "...",
      "oportunidades_comerciales": "...",
      "estrategias": {
        "Producto": ["...", "..."],
        "Precio": ["...", "..."],
        "Innovación": ["...", "..."],
        "Comunicación": ["...", "..."],
        "Rentabilidad": ["...", "..."]
      }
    }
  ]
}`;

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      temperature: 0.4,
      response_format: { type: 'json_object' },
      messages: [
        {
          role: 'system',
          content: 'Eres un analista comercial experto. Respondes únicamente en JSON válido en español.',
        },
        { role: 'user', content: prompt },
      ],
    }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`OpenAI API respondió con estado ${response.status}: ${errText.slice(0, 180)}`);
  }

  const jsonResp = await response.json();
  let content: string = jsonResp?.choices?.[0]?.message?.content || '';
  content = content.trim();
  if (content.startsWith('```')) {
    content = content.replace(/^```json?\s*/i, '').replace(/```$/, '').trim();
  }

  const parsed = JSON.parse(content);
  if (!parsed?.clusters || !Array.isArray(parsed.clusters)) {
    throw new Error('La respuesta JSON de OpenAI no contiene el arreglo "clusters".');
  }

  const profiles: Record<number, ClusterProfile> = {};
  const strategies: Record<number, FivePillarsStrategy> = {};

  for (const item of parsed.clusters) {
    const cid = Number(item.cluster_id);
    profiles[cid] = {
      clusterId: cid,
      nombreComercial: String(item.nombre_comercial || `Segmento ${cid}`),
      descripcionPerfil: String(item.descripcion_perfil || ''),
      caracteristicasDiferenciadoras: Array.isArray(item.caracteristicas_diferenciadoras)
        ? item.caracteristicas_diferenciadoras.map(String)
        : [],
      comportamientoPredominante: String(item.comportamiento_predominante || ''),
      necesidadesRelevantes: String(item.necesidades_relevantes || ''),
      oportunidadesComerciales: String(item.oportunidades_comerciales || ''),
    };

    const st = item.estrategias || {};
    strategies[cid] = {
      Producto: Array.isArray(st.Producto) ? st.Producto.map(String) : ['Sin recomendaciones registradas.'],
      Precio: Array.isArray(st.Precio) ? st.Precio.map(String) : ['Sin recomendaciones registradas.'],
      Innovación: Array.isArray(st['Innovación'])
        ? st['Innovación'].map(String)
        : Array.isArray(st.Innovacion)
          ? st.Innovacion.map(String)
          : ['Sin recomendaciones registradas.'],
      Comunicación: Array.isArray(st['Comunicación'])
        ? st['Comunicación'].map(String)
        : Array.isArray(st.Comunicacion)
          ? st.Comunicacion.map(String)
          : ['Sin recomendaciones registradas.'],
      Rentabilidad: Array.isArray(st.Rentabilidad)
        ? st.Rentabilidad.map(String)
        : ['Sin recomendaciones registradas.'],
    };
  }

  for (const c of centroids) {
    if (!profiles[c.clusterId] || !strategies[c.clusterId]) {
      throw new Error(`La respuesta de OpenAI omitió el Cluster ${c.clusterId}.`);
    }
  }

  return { profiles, strategies };
}

export function buildSelfContainedHtmlReport(params: {
  projectName: string;
  sector: string;
  fileName: string;
  originalRowCount: number;
  originalColCount: number;
  duplicatesRemoved: number;
  modeledRowCount: number;
  idCols: string[];
  numericCols: string[];
  imputationSummary: string[];
  businessRuleSummary: string[];
  scalingMethod: ScalingMethod;
  algorithm: ClusteringAlgorithm;
  k: number;
  metric: DistanceMetric;
  linkage: LinkageMethod;
  businessKNote: string;
  metrics: ClusterMetrics;
  centroids: CentroidRow[];
  labels: number[];
  pca: PCAProjection;
  profiles: Record<number, ClusterProfile>;
  strategies: Record<number, FivePillarsStrategy>;
  aiModeLabel: string;
}): string {
  const {
    projectName,
    sector,
    fileName,
    originalRowCount,
    originalColCount,
    duplicatesRemoved,
    modeledRowCount,
    idCols,
    numericCols,
    imputationSummary,
    businessRuleSummary,
    scalingMethod,
    algorithm,
    k,
    metric,
    linkage,
    businessKNote,
    metrics,
    centroids,
    labels,
    pca,
    profiles,
    strategies,
    aiModeLabel,
  } = params;

  const dateStr = new Date().toLocaleString('es-CO');

  // Construir traces PCA para Plotly embebido
  const pcaTraces = centroids.map((c) => {
    const cid = c.clusterId;
    const xVals: number[] = [];
    const yVals: number[] = [];
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] === cid) {
        xVals.push(pca.points[i][0]);
        yVals.push(pca.points[i][1]);
      }
    }
    const name = profiles[cid]?.nombreComercial || `Cluster ${cid}`;
    return {
      x: xVals,
      y: yVals,
      mode: 'markers',
      type: 'scatter',
      name: `${name} (n=${c.n})`,
      marker: { size: 7, color: CLUSTER_COLORS[cid % CLUSTER_COLORS.length], opacity: 0.72 },
    };
  });

  pcaTraces.push({
    x: pca.centroidPoints.map((p) => p[0]),
    y: pca.centroidPoints.map((p) => p[1]),
    mode: 'markers+text',
    type: 'scatter',
    name: 'Centroides',
    text: centroids.map((c) => `C${c.clusterId}`),
    textposition: 'top center',
    marker: {
      size: 15,
      symbol: 'diamond',
      color: '#FCD116',
      line: { width: 2, color: '#0F172A' },
    },
  } as never);

  // Normalizar radar 0-1
  const radarTraces = centroids.map((c) => {
    const vals = numericCols.map((col) => {
      const colVals = centroids.map((item) => item.values[col]);
      const minV = Math.min(...colVals);
      const maxV = Math.max(...colVals);
      return maxV - minV > 0 ? (c.values[col] - minV) / (maxV - minV) : 0.5;
    });
    vals.push(vals[0]);
    const theta = [...numericCols, numericCols[0]];
    const realText = [...numericCols.map((col) => `${col}: ${formatNum(c.values[col])}`), `${numericCols[0]}: ${formatNum(c.values[numericCols[0]])}`];
    const name = profiles[c.clusterId]?.nombreComercial || `Cluster ${c.clusterId}`;
    return {
      type: 'scatterpolar',
      r: vals,
      theta,
      fill: 'toself',
      opacity: 0.65,
      name,
      customdata: realText,
      hovertemplate: `<b>${name}</b><br>%{customdata}<br>Índice normalizado: %{r:.2f}<extra></extra>`,
      line: { color: CLUSTER_COLORS[c.clusterId % CLUSTER_COLORS.length], width: 2 },
    };
  });

  const centroidHeaderCols = numericCols.map((c) => `<th>${c}</th>`).join('');
  const centroidRowsHtml = centroids
    .map((c) => {
      const cells = numericCols.map((col) => `<td>${formatNum(c.values[col])}</td>`).join('');
      const name = profiles[c.clusterId]?.nombreComercial || `Cluster ${c.clusterId}`;
      return `<tr>
        <td><strong>Cluster ${c.clusterId}</strong><br><small style="color:#64748B;">${name}</small></td>
        ${cells}
        <td><strong>${c.n.toLocaleString('es-CO')}</strong></td>
        <td><strong>${c.percentage.toFixed(2)}%</strong></td>
      </tr>`;
    })
    .join('');

  const segmentsHtml = centroids
    .map((c) => {
      const p = profiles[c.clusterId];
      const st = strategies[c.clusterId];
      const color = CLUSTER_COLORS[c.clusterId % CLUSTER_COLORS.length];
      const pillars: (keyof FivePillarsStrategy)[] = [
        'Producto',
        'Precio',
        'Innovación',
        'Comunicación',
        'Rentabilidad',
      ];
      const pillarsHtml = pillars
        .map(
          (pillar) => `
          <div class="pilar">
            <h4>${pillar}</h4>
            <ul>${(st?.[pillar] || []).map((item) => `<li>${item}</li>`).join('')}</ul>
          </div>`
        )
        .join('');

      return `
      <article class="segmento-card" style="border-top: 4px solid ${color};">
        <div class="seg-header">
          <h3>Cluster ${c.clusterId}: ${p?.nombreComercial || `Segmento ${c.clusterId}`}</h3>
          <span class="seg-metrica">${c.n.toLocaleString('es-CO')} registros · ${c.percentage.toFixed(1)}%</span>
        </div>
        <p><strong>Descripción del perfil:</strong> ${p?.descripcionPerfil || ''}</p>
        <p><strong>Comportamiento predominante:</strong> ${p?.comportamientoPredominante || ''}</p>
        <p><strong>Necesidades relevantes:</strong> ${p?.necesidadesRelevantes || ''}</p>
        <p><strong>Oportunidades comerciales:</strong> ${p?.oportunidadesComerciales || ''}</p>
        <p><strong>Características diferenciadoras:</strong></p>
        <ul>${(p?.caracteristicasDiferenciadoras || []).map((d) => `<li>${d}</li>`).join('')}</ul>
        <h4 style="margin-top:16px; color:#0033A0;">Estrategia Comercial por Pilares</h4>
        <div class="pilares-grid">${pillarsHtml}</div>
      </article>`;
    })
    .join('');

  const largest = [...centroids].sort((a, b) => b.n - a.n)[0];
  const largestName = profiles[largest.clusterId]?.nombreComercial || `Cluster ${largest.clusterId}`;
  const var1 = (pca.explainedVarianceRatio[0] * 100).toFixed(1);
  const var2 = (pca.explainedVarianceRatio[1] * 100).toFixed(1);

  return `<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>${projectName} - Informe Ejecutivo</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #F8FAFC;
    color: #0F172A;
    margin: 0;
    padding: 0;
    line-height: 1.6;
  }
  .contenedor {
    max-width: 1120px;
    margin: 0 auto;
    padding: 32px 24px 64px 24px;
  }
  .portada {
    background: linear-gradient(125deg, #0033A0 0%, #002370 65%, #00B5AD 100%);
    color: #FFFFFF;
    padding: 40px 36px;
    border-radius: 12px;
    border-bottom: 6px solid #FCD116;
    margin-bottom: 24px;
  }
  .portada h1 { margin: 0 0 8px 0; font-size: 28px; }
  .portada p { margin: 4px 0; color: #E2E8F0; font-size: 14px; }
  .franja {
    display: flex; height: 6px; border-radius: 4px; overflow: hidden; margin-bottom: 28px;
  }
  .seccion {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 22px;
  }
  .seccion h2 {
    margin-top: 0;
    color: #0033A0;
    font-size: 19px;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 8px;
  }
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 14px;
    margin-top: 12px;
  }
  .kpi {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 12px 14px;
  }
  .kpi small { display: block; color: #64748B; font-size: 12px; font-weight: 600; }
  .kpi strong { font-size: 20px; color: #0F172A; font-variant-numeric: tabular-nums; }
  .tabla-datos {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 12px;
    font-variant-numeric: tabular-nums;
  }
  .tabla-datos th {
    background: #0033A0;
    color: #FFFFFF;
    text-align: left;
    padding: 10px 12px;
  }
  .tabla-datos td {
    border-bottom: 1px solid #E2E8F0;
    padding: 9px 12px;
  }
  .tabla-datos tr:nth-child(even) { background-color: #F8FAFC; }
  .segmento-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 22px 24px;
    margin-bottom: 20px;
  }
  .seg-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 10px;
    margin-bottom: 12px;
  }
  .seg-header h3 { margin: 0; color: #0F172A; font-size: 18px; }
  .seg-metrica { font-weight: 700; color: #0033A0; font-size: 14px; }
  .pilares-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 14px;
    margin-top: 12px;
  }
  .pilar {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 12px 14px;
  }
  .pilar h4 { margin: 0 0 6px 0; color: #00A99D; font-size: 14px; }
  .pilar ul { margin: 0; padding-left: 18px; font-size: 13px; }
</style>
</head>
<body>
<div class="contenedor">
  <header class="portada">
    <p>Clustering &amp; Commercial Strategy Lab · Informe Ejecutivo</p>
    <h1>${projectName}</h1>
    <p><strong>Sector económico:</strong> ${sector} · <strong>Archivo:</strong> ${fileName} · <strong>Generado:</strong> ${dateStr}</p>
  </header>

  <div class="franja">
    <div style="flex:1; background:#78BE20;"></div>
    <div style="flex:1; background:#FCD116;"></div>
    <div style="flex:1; background:#00B5AD;"></div>
    <div style="flex:1; background:#0033A0;"></div>
  </div>

  <section class="seccion">
    <h2>1. Descripción de la Base de Datos y Preparación</h2>
    <div class="kpi-grid">
      <div class="kpi"><small>Registros originales</small><strong>${originalRowCount.toLocaleString('es-CO')}</strong></div>
      <div class="kpi"><small>Columnas totales</small><strong>${originalColCount}</strong></div>
      <div class="kpi"><small>Duplicados eliminados</small><strong>${duplicatesRemoved.toLocaleString('es-CO')}</strong></div>
      <div class="kpi"><small>Registros modelados</small><strong>${modeledRowCount.toLocaleString('es-CO')}</strong></div>
    </div>
    <p style="margin-top:14px;"><strong>Identificadores excluidos:</strong> ${idCols.length ? idCols.join(', ') : 'Ninguno'}</p>
    <p><strong>Variables numéricas del modelo:</strong> ${numericCols.join(', ')}</p>
    <p><strong>Resumen de imputación:</strong></p>
    <ul>${imputationSummary.map((s) => `<li>${s}</li>`).join('')}</ul>
    <p><strong>Reglas de negocio aplicadas:</strong></p>
    <ul>${businessRuleSummary.map((s) => `<li>${s}</li>`).join('')}</ul>
  </section>

  <section class="seccion">
    <h2>2. Metodología de Clustering y Métricas de Evaluación</h2>
    <div class="kpi-grid">
      <div class="kpi"><small>Algoritmo</small><strong>${algorithm}</strong></div>
      <div class="kpi"><small>Clusters (K)</small><strong>${k}</strong></div>
      <div class="kpi"><small>Escalamiento</small><strong>${scalingMethod}</strong></div>
      <div class="kpi"><small>Silhouette Score</small><strong>${metrics.silhouette.toFixed(4)}</strong></div>
      <div class="kpi"><small>Davies-Bouldin</small><strong>${metrics.daviesBouldin.toFixed(4)}</strong></div>
      <div class="kpi"><small>Calinski-Harabasz</small><strong>${metrics.calinskiHarabasz.toFixed(2)}</strong></div>
    </div>
    <p style="margin-top:14px;">
      <strong>Distancia:</strong> ${metric}${algorithm !== 'K-Means' ? ` · <strong>Enlace:</strong> ${linkage}` : ''}
      ${businessKNote ? ` · <strong>Criterio de negocio:</strong> ${businessKNote}` : ''}
    </p>
  </section>

  <section class="seccion">
    <h2>3. Tabla de Centroides Desescalados (Unidades Originales)</h2>
    <div style="overflow-x:auto;">
      <table class="tabla-datos">
        <thead>
          <tr><th>Segmento</th>${centroidHeaderCols}<th>N</th><th>Porcentaje</th></tr>
        </thead>
        <tbody>${centroidRowsHtml}</tbody>
      </table>
    </div>
  </section>

  <section class="seccion">
    <h2>4. Visualización PCA 2D y Gráfico de Radar</h2>
    <p>El análisis PCA se utiliza únicamente como técnica de proyección visual en 2D sin sustituir las variables originales del modelo.</p>
    <div id="pca-chart" style="width:100%;height:480px;"></div>
    <div id="radar-chart" style="width:100%;height:480px;margin-top:20px;"></div>
  </section>

  <section class="seccion">
    <h2>5. Perfiles Comerciales y Estrategias por Segmento</h2>
    <p>Modalidad de generación: <strong>${aiModeLabel}</strong>. Las recomendaciones diferencian entre patrones observados y acciones sugeridas.</p>
    ${segmentsHtml}
  </section>

  <section class="seccion">
    <h2>6. Conclusiones Ejecutivas</h2>
    <p>
      El análisis sobre <strong>${modeledRowCount.toLocaleString('es-CO')}</strong> registros del sector
      <strong>${sector}</strong> permitió estructurar <strong>${k}</strong> segmentos claramente diferenciados.
      El grupo de mayor tamaño es <strong>${largestName}</strong> con ${largest.n.toLocaleString('es-CO')} registros
      (${largest.percentage.toFixed(1)}% del total).
    </p>
    <p>
      Se recomienda priorizar la ejecución de pilotos por segmento alineando las capacidades operativas y comerciales
      a los cinco pilares estratégicos definidos en este informe.
    </p>
  </section>
</div>
<script>
  Plotly.newPlot('pca-chart', ${JSON.stringify(pcaTraces)}, {
    title: 'Proyección PCA 2D de Segmentos y Centroides (Varianza explicada: ${(Number(var1) + Number(var2)).toFixed(1)}%)',
    xaxis: { title: 'Componente Principal 1 (${var1}%)' },
    yaxis: { title: 'Componente Principal 2 (${var2}%)' },
    template: 'plotly_white',
    legend: { orientation: 'h', y: -0.22 }
  }, { responsive: true });

  Plotly.newPlot('radar-chart', ${JSON.stringify(radarTraces)}, {
    title: 'Comparación Multivariada de Perfiles (Centroides Normalizados 0–1)',
    polar: { radialaxis: { visible: true, range: [0, 1.05] } },
    template: 'plotly_white',
    legend: { orientation: 'h', y: -0.22 }
  }, { responsive: true });
</script>
</body>
</html>`;
}
