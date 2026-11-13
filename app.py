# -*- coding: utf-8 -*-
"""
Clustering & Commercial Strategy Lab
Aplicación web interactiva en Python y Streamlit para el análisis de datos,
segmentación mediante Machine Learning (clustering) y generación de estrategias
comerciales accionables por segmento.
"""

import io
import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    from openai import OpenAI
    OPENAI_DISPONIBLE = True
except ImportError:
    OPENAI_DISPONIBLE = False


# ==============================================================================
# CONFIGURACIÓN GLOBAL, PALETA CORPORATIVA Y CONSTANTES
# ==============================================================================
COLORES_CORPORATIVOS = {
    "azul_rey": "#0033A0",
    "verde_biche": "#78BE20",
    "amarillo": "#FCD116",
    "azul_aguamarina": "#00B5AD",
    "oscuro": "#0F172A",
    "gris_texto": "#334155",
    "borde": "#E2E8F0",
    "fondo": "#F8FAFC",
}

PALETA_CLUSTERS = [
    "#0033A0",  # Azul Rey
    "#00B5AD",  # Azul Aguamarina
    "#78BE20",  # Verde Biche
    "#D97706",  # Ámbar / Amarillo oscuro legible
    "#6366F1",  # Índigo
    "#0D9488",  # Verde azulado
    "#DC2626",  # Carmesí
    "#475569",  # Pizarra
    "#2563EB",  # Azul medio
    "#65A30D",  # Verde oliva
]

SECTORES_ECONOMICOS = [
    "Retail y comercio",
    "Banca y servicios financieros",
    "Transporte y logística",
    "Telecomunicaciones",
    "Salud",
    "Manufactura e industria",
    "Educación",
    "Seguridad y defensa",
    "Alimentos y bebidas",
    "Otro",
]

PASOS_FLUJO = [
    (1, "1. Carga de archivos"),
    (2, "2. Vista previa y sector"),
    (3, "3. Inspección de variables"),
    (4, "4. Diagnóstico y duplicados"),
    (5, "5. Imputación de datos"),
    (6, "6. Análisis y reglas de negocio"),
    (7, "7. Número óptimo de clusters (K)"),
    (8, "8. Algoritmo y distancia"),
    (9, "9. Ejecución del clustering"),
    (10, "10. Centroides y visualización"),
    (11, "11. Perfilamiento comercial"),
    (12, "12. Estrategias (5 pilares)"),
    (13, "13. Exportación informe HTML"),
]


# ==============================================================================
# INICIALIZACIÓN DE ESTADO Y UTILIDADES DE CONTROL DE FLUJO
# ==============================================================================
def inicializar_estado() -> None:
    """Inicializa todas las llaves necesarias en st.session_state."""
    valores_por_defecto = {
        "paso_actual": 1,
        "nombre_proyecto": "Análisis de Segmentación Comercial",
        "nombre_archivo": None,
        "df_original": None,
        "df_trabajo": None,
        "df_limpio": None,
        "df_imputado": None,
        "df_filtrado": None,
        "df_resultado": None,
        "sector_economico": "Retail y comercio",
        "sector_personalizado": "",
        "cols_id": [],
        "cols_numericas": [],
        "duplicados_eliminados": 0,
        "resumen_imputacion": [],
        "reglas_negocio": {},
        "resumen_reglas": [],
        "metodo_escalamiento": "StandardScaler",
        "resultados_k": None,
        "k_seleccionado": 4,
        "criterio_negocio_nota": "",
        "algoritmo": "K-Means",
        "metrica_distancia": "euclidean",
        "metodo_enlace": "ward",
        "kmeans_n_init": 10,
        "kmeans_random_state": 42,
        "kmeans_max_iter": 300,
        "kmeans_tol": 1e-4,
        "metricas_modelo": {},
        "centroides_df": None,
        "pca_coords": None,
        "pca_varianza": None,
        "perfiles_clusters": None,
        "estrategias_clusters": None,
        "modo_ia_usado": "Simulación / Plantilla",
        "openai_model": "gpt-4o-mini",
    }
    for clave, valor in valores_por_defecto.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor


def invalidar_desde_paso(paso: int) -> None:
    """
    Invalida de manera controlada únicamente los resultados posteriores
    cuando el usuario modifica decisiones en un paso anterior.
    """
    if paso <= 1:
        st.session_state.df_trabajo = None
        st.session_state.cols_id = []
        st.session_state.cols_numericas = []
    if paso <= 3:
        st.session_state.df_limpio = None
        st.session_state.duplicados_eliminados = 0
    if paso <= 4:
        st.session_state.df_imputado = None
        st.session_state.resumen_imputacion = []
    if paso <= 5:
        st.session_state.df_filtrado = None
        st.session_state.reglas_negocio = {}
        st.session_state.resumen_reglas = []
    if paso <= 6:
        st.session_state.resultados_k = None
    if paso <= 8:
        st.session_state.df_resultado = None
        st.session_state.metricas_modelo = {}
        st.session_state.centroides_df = None
        st.session_state.pca_coords = None
        st.session_state.pca_varianza = None
    if paso <= 10:
        st.session_state.perfiles_clusters = None
        st.session_state.estrategias_clusters = None


def obtener_sector_efectivo() -> str:
    """Retorna el nombre completo del sector económico seleccionado."""
    sector = st.session_state.get("sector_economico", "Retail y comercio")
    if sector == "Otro":
        personalizado = st.session_state.get("sector_personalizado", "").strip()
        return personalizado if personalizado else "Otro sector económico"
    return sector


def obtener_df_para_modelar() -> Optional[pd.DataFrame]:
    """Retorna el DataFrame más avanzado disponible antes del clustering."""
    for clave in ["df_filtrado", "df_imputado", "df_limpio", "df_trabajo", "df_original"]:
        df = st.session_state.get(clave)
        if df is not None:
            return df
    return None


def paso_disponible(paso: int) -> bool:
    """Determina si el usuario tiene los requisitos mínimos para acceder a un paso."""
    if paso == 1:
        return True
    if st.session_state.df_original is None:
        return False
    if paso <= 3:
        return True
    if len(st.session_state.cols_numericas) < 2:
        return False
    if paso <= 6:
        return True
    if st.session_state.df_filtrado is None and st.session_state.df_imputado is None:
        return False
    if paso <= 9:
        return True
    if st.session_state.df_resultado is None:
        return False
    if paso <= 11:
        return True
    if st.session_state.perfiles_clusters is None:
        return False
    return True


def generar_dataset_ejemplo() -> pd.DataFrame:
    """Genera una base empresarial realista de clientes para pruebas inmediatas."""
    rng = np.random.default_rng(42)
    n = 180
    segmentos = rng.choice([0, 1, 2, 3], size=n, p=[0.30, 0.30, 0.25, 0.15])

    ids = [f"CLI-{1000 + i}" for i in range(n)]
    edades = np.where(
        segmentos == 0, rng.normal(28, 5, n),
        np.where(segmentos == 1, rng.normal(42, 6, n),
                 np.where(segmentos == 2, rng.normal(54, 7, n), rng.normal(35, 8, n)))
    ).round(0)
    ingresos = np.where(
        segmentos == 0, rng.normal(2800, 500, n),
        np.where(segmentos == 1, rng.normal(6200, 900, n),
                 np.where(segmentos == 2, rng.normal(9500, 1200, n), rng.normal(4100, 700, n)))
    ).round(2)
    frecuencia = np.where(
        segmentos == 0, rng.normal(4, 1.5, n),
        np.where(segmentos == 1, rng.normal(12, 2.5, n),
                 np.where(segmentos == 2, rng.normal(22, 4, n), rng.normal(2, 1, n)))
    ).clip(1, 40).round(0)
    ticket_promedio = np.where(
        segmentos == 0, rng.normal(65, 18, n),
        np.where(segmentos == 1, rng.normal(185, 35, n),
                 np.where(segmentos == 2, rng.normal(410, 70, n), rng.normal(95, 25, n)))
    ).clip(15, 900).round(2)
    antiguedad_meses = np.where(
        segmentos == 0, rng.normal(10, 4, n),
        np.where(segmentos == 1, rng.normal(32, 8, n),
                 np.where(segmentos == 2, rng.normal(64, 12, n), rng.normal(6, 3, n)))
    ).clip(1, 120).round(0)

    canales = rng.choice(["Tienda Física", "E-Commerce", "App Móvil", "Mixto"], size=n)
    ciudades = rng.choice(["Bogotá", "Medellín", "Cali", "Barranquilla"], size=n)

    df = pd.DataFrame({
        "ID_Cliente": ids,
        "Edad": edades,
        "Ingreso_Mensual_USD": ingresos,
        "Frecuencia_Compra_Anual": frecuencia,
        "Ticket_Promedio_USD": ticket_promedio,
        "Antiguedad_Meses": antiguedad_meses,
        "Canal_Preferido": canales,
        "Ciudad": ciudades,
    })

    # Insertar algunos valores faltantes, duplicados y atípicos realistas para probar limpieza
    df.loc[3, "Ingreso_Mensual_USD"] = np.nan
    df.loc[18, "Ingreso_Mensual_USD"] = np.nan
    df.loc[27, "Ticket_Promedio_USD"] = np.nan
    df.loc[45, "Canal_Preferido"] = np.nan
    df.loc[10, "Edad"] = 145  # Atípico de regla de negocio
    df.loc[75, "Ticket_Promedio_USD"] = -25.0  # Valor negativo inválido
    df = pd.concat([df, df.iloc[[0, 5, 12]]], ignore_index=True)
    return df


# ==============================================================================
# ESTILOS CSS PERSONALIZADOS (PALETA CORPORATIVA)
# ==============================================================================
def inyectar_estilos() -> None:
    """Aplica estilos corporativos limpios y profesionales en Streamlit."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {COLORES_CORPORATIVOS["fondo"]};
        }}
        .banner-institucional {{
            background: linear-gradient(120deg, {COLORES_CORPORATIVOS["azul_rey"]} 0%, #002370 65%, {COLORES_CORPORATIVOS["azul_aguamarina"]} 100%);
            padding: 1.4rem 1.8rem;
            border-radius: 10px;
            color: #FFFFFF;
            margin-bottom: 1.25rem;
            border-bottom: 4px solid {COLORES_CORPORATIVOS["amarillo"]};
        }}
        .banner-institucional h1 {{
            margin: 0;
            font-size: 1.65rem;
            font-weight: 700;
            color: #FFFFFF;
        }}
        .banner-institucional p {{
            margin: 0.35rem 0 0 0;
            font-size: 0.92rem;
            color: #E2E8F0;
        }}
        .kpi-card {{
            background-color: #FFFFFF;
            border: 1px solid {COLORES_CORPORATIVOS["borde"]};
            border-top: 3px solid {COLORES_CORPORATIVOS["azul_rey"]};
            border-radius: 8px;
            padding: 0.9rem 1rem;
        }}
        .kpi-label {{
            font-size: 0.78rem;
            color: #64748B;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }}
        .kpi-value {{
            font-size: 1.45rem;
            font-weight: 700;
            color: {COLORES_CORPORATIVOS["oscuro"]};
        }}
        .pilar-box {{
            background-color: #FFFFFF;
            border: 1px solid {COLORES_CORPORATIVOS["borde"]};
            border-radius: 8px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.8rem;
        }}
        .pilar-titulo {{
            font-weight: 700;
            color: {COLORES_CORPORATIVOS["azul_rey"]};
            margin-bottom: 0.4rem;
            font-size: 0.98rem;
        }}
        .franja-colores {{
            display: flex;
            height: 5px;
            width: 100%;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# FUNCIONES DE CÁLCULO ANALÍTICO Y MACHINE LEARNING
# ==============================================================================
def cargar_archivo_multiformato(
    archivo: Any, separador: str, codificacion: str
) -> pd.DataFrame:
    """Lee archivos CSV, TXT, XLS y XLSX con manejo de codificación y excepciones."""
    nombre = archivo.name.lower()
    contenido = archivo.getvalue()
    if not contenido or len(contenido) == 0:
        raise ValueError("El archivo cargado está completamente vacío.")

    buffer = io.BytesIO(contenido)

    if nombre.endswith((".csv", ".txt")):
        codificaciones_a_probar = [codificacion, "utf-8", "latin-1", "cp1252"]
        ultimo_error = None
        for enc in dict.fromkeys(codificaciones_a_probar):
            try:
                buffer.seek(0)
                df = pd.read_csv(buffer, sep=separador, encoding=enc)
                if df.empty or df.shape[1] == 0:
                    raise ValueError("El archivo no contiene registros ni columnas válidas.")
                return df
            except UnicodeDecodeError as err:
                ultimo_error = err
                continue
        raise ValueError(f"No fue posible decodificar el archivo con {codificacion}. Detalle: {ultimo_error}")

    if nombre.endswith(".xlsx"):
        df = pd.read_excel(buffer, engine="openpyxl")
    elif nombre.endswith(".xls"):
        df = pd.read_excel(buffer, engine="xlrd")
    else:
        raise ValueError("Formato no compatible. Utilice archivos CSV, TXT, XLS o XLSX.")

    if df.empty or df.shape[1] == 0:
        raise ValueError("La hoja de cálculo no contiene datos válidos.")
    return df


def construir_tabla_inspeccion(df: pd.DataFrame) -> pd.DataFrame:
    """Genera tabla de inspección: Variable, Tipo de dato, Valores nulos, Valores únicos."""
    filas = []
    for col in df.columns:
        filas.append({
            "Variable": str(col),
            "Tipo de dato": str(df[col].dtype),
            "Valores nulos": int(df[col].isna().sum()),
            "Valores únicos": int(df[col].nunique(dropna=True)),
        })
    return pd.DataFrame(filas)


def calcular_matriz_escalada(
    df: pd.DataFrame, columnas: List[str], metodo: str
) -> Tuple[np.ndarray, Any]:
    """Escala las columnas numéricas mediante StandardScaler o MinMaxScaler."""
    datos = df[columnas].copy()
    if datos.isna().any().any():
        raise ValueError(
            "Existen valores faltantes en las variables seleccionadas. "
            "Complete el paso de imputación antes de modelar."
        )
    scaler = StandardScaler() if metodo == "StandardScaler" else MinMaxScaler()
    matriz_escalada = scaler.fit_transform(datos.values)
    return matriz_escalada, scaler


def evaluar_rango_k(matriz_escalada: np.ndarray, k_max: int = 10) -> pd.DataFrame:
    """Calcula Inercia (Codo) y Silhouette Score para K desde 2 hasta k_max."""
    n_registros = matriz_escalada.shape[0]
    limite_superior = min(k_max, n_registros - 1)
    resultados = []
    for k in range(2, limite_superior + 1):
        modelo = KMeans(n_clusters=k, n_init=10, random_state=42)
        etiquetas = modelo.fit_predict(matriz_escalada)
        inercia = float(modelo.inertia_)
        silueta = float(silhouette_score(matriz_escalada, etiquetas))
        resultados.append({
            "K": k,
            "Inercia": round(inercia, 4),
            "Silhouette": round(silueta, 4),
        })
    return pd.DataFrame(resultados)


def recomendar_k_tecnico(df_k: pd.DataFrame) -> Tuple[int, int]:
    """
    Identifica el mejor K por Silhouette Score y estima el punto de inflexión
    (codo) mediante la distancia máxima a la recta entre los extremos de la curva.
    """
    mejor_sil_k = int(df_k.loc[df_k["Silhouette"].idxmax(), "K"])

    ks = df_k["K"].values.astype(float)
    inercias = df_k["Inercia"].values.astype(float)
    if len(ks) >= 3:
        p1 = np.array([ks[0], inercias[0]])
        p2 = np.array([ks[-1], inercias[-1]])
        vec_linea = p2 - p1
        norma = np.linalg.norm(vec_linea)
        if norma > 0:
            distancias = [
                abs(np.cross(vec_linea, p1 - np.array([k_val, in_val]))) / norma
                for k_val, in_val in zip(ks, inercias)
            ]
            codo_k = int(ks[int(np.argmax(distancias))])
        else:
            codo_k = mejor_sil_k
    else:
        codo_k = mejor_sil_k

    return mejor_sil_k, codo_k


def validar_jerarquico(metrica: str, enlace: str) -> Tuple[bool, str]:
    """Valida compatibilidad entre métrica de distancia y método de enlace en Scikit-Learn."""
    if enlace == "ward" and metrica != "euclidean":
        return (
            False,
            "El método de enlace 'Ward' requiere obligatoriamente distancia 'Euclidiana' "
            "(sklearn.cluster.AgglomerativeClustering no admite Manhattan o Coseno con Ward).",
        )
    return True, "Combinación válida."


# ==============================================================================
# PERFILAMIENTO Y ESTRATEGIAS (MODO OPENAI Y MODO SIMULACIÓN)
# ==============================================================================
def generar_perfiles_simulacion(
    centroides_df: pd.DataFrame,
    cols_numericas: List[str],
    df_modelo: pd.DataFrame,
    sector: str,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, List[str]]]]:
    """
    Genera perfiles comerciales y estrategias de 5 pilares mediante reglas
    estructuradas basadas en las desviaciones de cada centroide respecto al promedio global.
    """
    medias_globales = df_modelo[cols_numericas].mean()
    stds_globales = df_modelo[cols_numericas].std().replace(0, 1.0)

    perfiles: Dict[int, Dict[str, Any]] = {}
    estrategias: Dict[int, Dict[str, List[str]]] = {}

    nombres_base = [
        "Consolidado de Alto Desempeño",
        "Expansivo en Desarrollo",
        "Operación Estandarizada",
        "Nicho Especializado",
        "Sensible a Eficiencia",
        "Tracción Moderada",
        "Emergente Focalizado",
        "Base Tradicional",
        "Alto Volumen Selectivo",
        "Segmento Conservador",
    ]

    for idx, fila in centroides_df.iterrows():
        cid = int(fila["Cluster"])
        n_seg = int(fila["N"])
        pct_seg = float(fila["Porcentaje"])

        z_scores = {}
        for col in cols_numericas:
            z_scores[col] = float((fila[col] - medias_globales[col]) / stds_globales[col])

        vars_ordenadas = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)
        var_max, z_max = vars_ordenadas[0]
        var_min, z_min = vars_ordenadas[-1]

        altas = [f"{v} ({fila[v]:,.2f}, +{z:.1f}σ)" for v, z in vars_ordenadas if z >= 0.35]
        bajas = [f"{v} ({fila[v]:,.2f}, {z:.1f}σ)" for v, z in vars_ordenadas if z <= -0.35]

        nivel_z_promedio = float(np.mean(list(z_scores.values())))
        if nivel_z_promedio > 0.45:
            arquetipo = "Líder de Alto Valor"
        elif nivel_z_promedio < -0.45:
            arquetipo = "Conservador / Contención"
        elif z_max > 0.7:
            arquetipo = f"Especializado en {var_max}"
        else:
            arquetipo = nombres_base[cid % len(nombres_base)]

        nombre_comercial = f"{arquetipo} · C{cid}"

        desc_altas = ", ".join(altas) if altas else f"valores cercanos a la media en {var_max}"
        desc_bajas = ", ".join(bajas) if bajas else f"sin rezagos marcados (menor en {var_min})"

        perfiles[cid] = {
            "nombre_comercial": nombre_comercial,
            "descripcion_perfil": (
                f"Segmento que agrupa {n_seg} registros ({pct_seg:.1f}% del total analizado) "
                f"en el sector {sector}. Se distingue estadísticamente por presentar niveles "
                f"superiores en {desc_altas} y valores más contenidos en {desc_bajas}."
            ),
            "caracteristicas_diferenciadoras": [
                f"Variable más sobresaliente: {var_max} con media de {fila[var_max]:,.2f} (z = {z_max:+.2f}).",
                f"Variable más baja del perfil: {var_min} con media de {fila[var_min]:,.2f} (z = {z_min:+.2f}).",
                f"Representa el {pct_seg:.1f}% de la base operativa modelada ({n_seg} registros).",
            ],
            "comportamiento_predominante": (
                f"Patrón concentrado en torno a {var_max} ({fila[var_max]:,.2f}), "
                f"manteniendo una estructura equilibrada frente al comportamiento observado en {sector}."
            ),
            "necesidades_relevantes": (
                f"Requiere soluciones ajustadas a su nivel actual en {var_max} y acompañamiento "
                f"para fortalecer el desempeño en {var_min} sin sobredimensionar la oferta."
            ),
            "oportunidades_comerciales": (
                f"Estructurar ofertas diferenciadas aprovechando la tracción observada en {var_max} "
                f"y diseñar estímulos graduales orientados a cerrar brechas en {var_min}."
            ),
        }

        estrategias[cid] = {
            "Producto": [
                f"Priorizar portafolio principal alineado con el desempeño de {var_max} (media {fila[var_max]:,.2f}) dentro de {sector}.",
                f"Diseñar paquetes modulares que combinen la oferta central con componentes de entrada para impulsar {var_min}.",
                "Implementar esquemas de venta cruzada basados en los atributos predominantes del segmento.",
            ],
            "Precio": [
                f"Calibrar la estructura tarifaria según el volumen observado en {var_max} ({fila[var_max]:,.2f}).",
                "Evitar descuentos generalizados si el segmento ya muestra alta tracción; emplear incentivos por cumplimiento o permanencia.",
                f"Evaluar planes escalonados para facilitar la adopción en registros con menor indicador de {var_min}.",
            ],
            "Innovación": [
                f"Automatizar flujos de atención y seguimiento operativo específicamente para este grupo ({pct_seg:.1f}% de la base).",
                f"Incorporar tableros de alerta temprana sobre variaciones en {var_max} y {var_min}.",
                f"Pilotar servicios digitales o de autogestión adaptados a la escala del segmento en {sector}.",
            ],
            "Comunicación": [
                f"Construir mensajes comerciales centrados en los beneficios tangibles asociados a {var_max}.",
                "Seleccionar canales digitales y directos proporcionales al tamaño del segmento y capacidad operativa.",
                "Establecer campañas de fidelización y reactivación diferenciadas frente a los demás clusters.",
            ],
            "Rentabilidad": [
                f"Asignar capacidad comercial de manera proporcional al peso del segmento ({pct_seg:.1f}% de los registros).",
                "Reducir costos de atención mediante estandarización en variables de menor dispersión.",
                f"Monitorear periódicamente la evolución de {var_max} y {var_min} antes de comprometer inversiones mayores.",
            ],
        }

    return perfiles, estrategias


def generar_perfiles_openai(
    api_key: str,
    modelo: str,
    centroides_df: pd.DataFrame,
    cols_numericas: List[str],
    sector: str,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, List[str]]]]:
    """
    Invoca la API de OpenAI solicitando JSON estructurado con perfiles
    comerciales y los 5 pilares estratégicos para cada cluster.
    """
    if not OPENAI_DISPONIBLE:
        raise RuntimeError("La librería 'openai' no está instalada en el entorno.")

    client = OpenAI(api_key=api_key)
    tabla_json = centroides_df.to_dict(orient="records")

    prompt = f"""
Actúa como un Director de Estrategia Comercial y Científico de Datos Senior.
Debes interpretar los resultados de un modelo de clustering para el sector económico: "{sector}".

Número total de clusters: {len(centroides_df)}
Variables numéricas utilizadas: {', '.join(cols_numericas)}

Tabla de centroides desescalados (medias reales en unidades originales) junto con N (cantidad) y Porcentaje (%):
{json.dumps(tabla_json, ensure_ascii=False, indent=2)}

Reglas analíticas estrictas:
1. Diferencia claramente entre hallazgos observables en los centroides y recomendaciones comerciales.
2. No inventes cifras financieras ni afirmaciones causales que no puedan sustentarse en los datos.
3. Contextualiza cada recomendación al sector "{sector}".
4. Cada cluster debe tener recomendaciones claramente diferenciadas.
5. Responde ÚNICAMENTE con un objeto JSON válido con la estructura exacta:
{{
  "clusters": [
    {{
      "cluster_id": 0,
      "nombre_comercial": "...",
      "descripcion_perfil": "...",
      "caracteristicas_diferenciadoras": ["...", "...", "..."],
      "comportamiento_predominante": "...",
      "necesidades_relevantes": "...",
      "oportunidades_comerciales": "...",
      "estrategias": {{
        "Producto": ["...", "..."],
        "Precio": ["...", "..."],
        "Innovación": ["...", "..."],
        "Comunicación": ["...", "..."],
        "Rentabilidad": ["...", "..."]
      }}
    }}
  ]
}}
"""

    respuesta = client.chat.completions.create(
        model=modelo,
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Eres un analista comercial experto. Respondes exclusivamente en JSON válido en español.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    contenido = respuesta.choices[0].message.content or ""
    contenido = contenido.strip()
    if contenido.startswith("```"):
        contenido = contenido.strip("`").replace("json", "", 1).strip()

    data = json.loads(contenido)
    if "clusters" not in data or not isinstance(data["clusters"], list):
        raise ValueError("La respuesta de IA no contiene la lista 'clusters' esperada.")

    perfiles: Dict[int, Dict[str, Any]] = {}
    estrategias: Dict[int, Dict[str, List[str]]] = {}
    pilares_obligatorios = ["Producto", "Precio", "Innovación", "Comunicación", "Rentabilidad"]

    for item in data["clusters"]:
        cid = int(item["cluster_id"])
        perfiles[cid] = {
            "nombre_comercial": str(item.get("nombre_comercial", f"Segmento {cid}")),
            "descripcion_perfil": str(item.get("descripcion_perfil", "")),
            "caracteristicas_diferenciadoras": list(item.get("caracteristicas_diferenciadoras", [])),
            "comportamiento_predominante": str(item.get("comportamiento_predominante", "")),
            "necesidades_relevantes": str(item.get("necesidades_relevantes", "")),
            "oportunidades_comerciales": str(item.get("oportunidades_comerciales", "")),
        }
        est_raw = item.get("estrategias", {})
        estrategias[cid] = {
            pilar: list(est_raw.get(pilar, ["Sin recomendaciones registradas."]))
            for pilar in pilares_obligatorios
        }

    # Validar cobertura completa de todos los clusters
    ids_esperados = [int(x) for x in centroides_df["Cluster"].tolist()]
    for cid in ids_esperados:
        if cid not in perfiles or cid not in estrategias:
            raise ValueError(f"La respuesta de OpenAI omitió el cluster {cid}.")

    return perfiles, estrategias


# ==============================================================================
# GRÁFICOS PLOTLY REUTILIZABLES (INTERFAZ Y EXPORTACIÓN HTML)
# ==============================================================================
def crear_figura_pca(
    pca_coords: np.ndarray,
    etiquetas: np.ndarray,
    varianza: np.ndarray,
    centroides_df: pd.DataFrame,
    cols_numericas: List[str],
    pca_modelo: PCA,
    scaler_modelo: Any,
    perfiles: Optional[Dict[int, Dict[str, Any]]] = None,
) -> go.Figure:
    """Construye gráfico PCA 2D interactivo con registros y centroides destacados."""
    fig = go.Figure()
    clusters_unicos = sorted(np.unique(etiquetas).tolist())

    for cid in clusters_unicos:
        mask = etiquetas == cid
        color = PALETA_CLUSTERS[cid % len(PALETA_CLUSTERS)]
        nombre = (
            perfiles[cid]["nombre_comercial"]
            if perfiles and cid in perfiles
            else f"Cluster {cid}"
        )
        fig.add_trace(
            go.Scatter(
                x=pca_coords[mask, 0],
                y=pca_coords[mask, 1],
                mode="markers",
                name=f"{nombre} (n={int(mask.sum())})",
                marker=dict(size=8, color=color, opacity=0.72, line=dict(width=0.6, color="#FFFFFF")),
                hovertemplate=f"<b>{nombre}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>",
            )
        )

    # Proyectar los centroides reales al mismo espacio PCA para visualización coherente
    centroides_escalados = scaler_modelo.transform(centroides_df[cols_numericas].values)
    centroides_pca = pca_modelo.transform(centroides_escalados)

    fig.add_trace(
        go.Scatter(
            x=centroides_pca[:, 0],
            y=centroides_pca[:, 1],
            mode="markers+text",
            name="Centroides",
            text=[f"C{int(c)}" for c in centroides_df["Cluster"]],
            textposition="top center",
            textfont=dict(size=12, color="#0F172A"),
            marker=dict(
                size=16,
                symbol="diamond",
                color=COLORES_CORPORATIVOS["amarillo"],
                line=dict(width=2, color="#0F172A"),
            ),
            hovertemplate="<b>Centroide %{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
        )
    )

    var_1 = varianza[0] * 100
    var_2 = varianza[1] * 100 if len(varianza) > 1 else 0.0
    fig.update_layout(
        title=f"Proyección PCA 2D de Segmentos y Centroides (Varianza explicada: {var_1 + var_2:.1f}%)",
        xaxis_title=f"Componente Principal 1 ({var_1:.1f}%)",
        yaxis_title=f"Componente Principal 2 ({var_2:.1f}%)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        height=500,
        margin=dict(l=50, r=30, t=60, b=90),
    )
    return fig


def crear_figura_radar(
    centroides_df: pd.DataFrame,
    cols_numericas: List[str],
    perfiles: Optional[Dict[int, Dict[str, Any]]] = None,
) -> go.Figure:
    """
    Construye gráfico de radar normalizando los centroides a [0, 1] entre clusters
    para permitir comparación multivariada manteniendo valores reales en hover.
    """
    fig = go.Figure()
    valores = centroides_df[cols_numericas].copy()
    mins = valores.min()
    maxs = valores.max()
    rangos = (maxs - mins).replace(0, 1.0)
    norm_df = (valores - mins) / rangos

    categorias = cols_numericas + [cols_numericas[0]]

    for idx, fila in centroides_df.iterrows():
        cid = int(fila["Cluster"])
        color = PALETA_CLUSTERS[cid % len(PALETA_CLUSTERS)]
        nombre = (
            perfiles[cid]["nombre_comercial"]
            if perfiles and cid in perfiles
            else f"Cluster {cid}"
        )
        r_vals = norm_df.loc[idx, cols_numericas].tolist()
        r_vals.append(r_vals[0])

        reales = [f"{col}: {fila[col]:,.2f}" for col in cols_numericas]
        reales.append(reales[0])

        fig.add_trace(
            go.Scatterpolar(
                r=r_vals,
                theta=categorias,
                fill="toself",
                opacity=0.65,
                name=nombre,
                line=dict(color=color, width=2),
                customdata=reales,
                hovertemplate=f"<b>{nombre}</b><br>%{{customdata}}<br>Índice relativo: %{{r:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Comparación Multivariada de Perfiles (Centroides Normalizados 0–1)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.05])),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=500,
        margin=dict(l=60, r=60, t=70, b=90),
    )
    return fig


# ==============================================================================
# GENERACIÓN DE INFORME EJECUTIVO HTML AUTOCONTENIDO
# ==============================================================================
def generar_html_informe_ejecutivo(
    fig_pca: go.Figure,
    fig_radar: go.Figure,
) -> str:
    """Construye un único archivo HTML autocontenido con las 16 secciones requeridas."""
    nombre_proyecto = st.session_state.nombre_proyecto
    sector = obtener_sector_efectivo()
    df_orig = st.session_state.df_original
    df_res = st.session_state.df_resultado
    cols_num = st.session_state.cols_numericas
    cols_id = st.session_state.cols_id
    metricas = st.session_state.metricas_modelo
    centroides_df = st.session_state.centroides_df
    perfiles = st.session_state.perfiles_clusters or {}
    estrategias = st.session_state.estrategias_clusters or {}

    fecha_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Embeber Plotly JS una sola vez inline para garantizar funcionamiento sin internet ni archivos externos
    html_pca = fig_pca.to_html(full_html=False, include_plotlyjs=True)
    html_radar = fig_radar.to_html(full_html=False, include_plotlyjs=False)

    tabla_centroides_html = centroides_df.to_html(
        index=False, classes="tabla-datos", float_format=lambda x: f"{x:,.2f}"
    )

    resumen_imp = st.session_state.resumen_imputacion or ["Sin imputaciones aplicadas."]
    resumen_reg = st.session_state.resumen_reglas or ["Sin exclusiones por reglas de negocio."]

    tarjetas_segmentos = []
    for _, fila in centroides_df.iterrows():
        cid = int(fila["Cluster"])
        n_seg = int(fila["N"])
        pct_seg = float(fila["Porcentaje"])
        perf = perfiles.get(cid, {})
        est = estrategias.get(cid, {})
        color = PALETA_CLUSTERS[cid % len(PALETA_CLUSTERS)]

        bloques_pilares = ""
        for pilar in ["Producto", "Precio", "Innovación", "Comunicación", "Rentabilidad"]:
            items = "".join(f"<li>{acc}</li>" for acc in est.get(pilar, []))
            bloques_pilares += f"""
            <div class="pilar">
                <h4>{pilar}</h4>
                <ul>{items}</ul>
            </div>
            """

        diferenciadores = "".join(
            f"<li>{d}</li>" for d in perf.get("caracteristicas_diferenciadoras", [])
        )

        tarjetas_segmentos.append(f"""
        <article class="segmento-card" style="border-top: 4px solid {color};">
            <div class="seg-header">
                <h3>Cluster {cid}: {perf.get('nombre_comercial', f'Segmento {cid}')}</h3>
                <span class="seg-metrica">{n_seg} registros · {pct_seg:.1f}%</span>
            </div>
            <p><strong>Descripción del perfil:</strong> {perf.get('descripcion_perfil', '')}</p>
            <p><strong>Comportamiento predominante:</strong> {perf.get('comportamiento_predominante', '')}</p>
            <p><strong>Necesidades relevantes:</strong> {perf.get('necesidades_relevantes', '')}</p>
            <p><strong>Oportunidades comerciales:</strong> {perf.get('oportunidades_comerciales', '')}</p>
            <p><strong>Características diferenciadoras:</strong></p>
            <ul>{diferenciadores}</ul>
            <h4 style="margin-top:16px; color:#0033A0;">Estrategia Comercial (5 Pilares)</h4>
            <div class="pilares-grid">{bloques_pilares}</div>
        </article>
        """)

    segmento_mayor = centroides_df.sort_values("N", ascending=False).iloc[0]
    cid_mayor = int(segmento_mayor["Cluster"])
    nombre_mayor = perfiles.get(cid_mayor, {}).get("nombre_comercial", f"Cluster {cid_mayor}")

    sil_texto = (
        f"{metricas.get('silhouette'):.4f}"
        if metricas.get("silhouette") is not None
        else "No aplicable"
    )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{nombre_proyecto} - Informe Ejecutivo</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #F8FAFC;
    color: #0F172A;
    margin: 0;
    padding: 0;
    line-height: 1.6;
  }}
  .contenedor {{
    max-width: 1120px;
    margin: 0 auto;
    padding: 32px 24px 64px 24px;
  }}
  .portada {{
    background: linear-gradient(125deg, #0033A0 0%, #002370 65%, #00B5AD 100%);
    color: #FFFFFF;
    padding: 40px 36px;
    border-radius: 12px;
    border-bottom: 6px solid #FCD116;
    margin-bottom: 28px;
  }}
  .portada h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
  .portada p {{ margin: 4px 0; color: #E2E8F0; font-size: 14px; }}
  .franja {{
    display: flex; height: 6px; border-radius: 4px; overflow: hidden; margin-bottom: 28px;
  }}
  .seccion {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 22px;
  }}
  .seccion h2 {{
    margin-top: 0;
    color: #0033A0;
    font-size: 19px;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 8px;
  }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 14px;
    margin-top: 12px;
  }}
  .kpi {{
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 12px 14px;
  }}
  .kpi small {{ display: block; color: #64748B; font-size: 12px; font-weight: 600; }}
  .kpi strong {{ font-size: 20px; color: #0F172A; }}
  .tabla-datos {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 12px;
  }}
  .tabla-datos th {{
    background: #0033A0;
    color: #FFFFFF;
    text-align: left;
    padding: 10px 12px;
  }}
  .tabla-datos td {{
    border-bottom: 1px solid #E2E8F0;
    padding: 9px 12px;
  }}
  .tabla-datos tr:nth-child(even) {{ background-color: #F8FAFC; }}
  .segmento-card {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 22px 24px;
    margin-bottom: 20px;
  }}
  .seg-header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 10px;
    margin-bottom: 12px;
  }}
  .seg-header h3 {{ margin: 0; color: #0F172A; font-size: 18px; }}
  .seg-metrica {{ font-weight: 700; color: #0033A0; font-size: 14px; }}
  .pilares-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 14px;
    margin-top: 12px;
  }}
  .pilar {{
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 12px 14px;
  }}
  .pilar h4 {{ margin: 0 0 6px 0; color: #00B5AD; font-size: 14px; }}
  .pilar ul {{ margin: 0; padding-left: 18px; font-size: 13px; }}
  @media print {{
    body {{ background: #FFFFFF; }}
    .seccion, .segmento-card {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="contenedor">
  <header class="portada">
    <p>Clustering &amp; Commercial Strategy Lab · Informe Ejecutivo</p>
    <h1>{nombre_proyecto}</h1>
    <p><strong>Sector económico:</strong> {sector} · <strong>Archivo analizado:</strong> {st.session_state.nombre_archivo or 'Base de ejemplo'} · <strong>Generado:</strong> {fecha_str}</p>
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
      <div class="kpi"><small>Registros originales</small><strong>{df_orig.shape[0]:,}</strong></div>
      <div class="kpi"><small>Columnas totales</small><strong>{df_orig.shape[1]}</strong></div>
      <div class="kpi"><small>Duplicados eliminados</small><strong>{st.session_state.duplicados_eliminados:,}</strong></div>
      <div class="kpi"><small>Registros modelados</small><strong>{df_res.shape[0]:,}</strong></div>
    </div>
    <p style="margin-top:14px;"><strong>Variables identificadoras excluidas:</strong> {', '.join(cols_id) if cols_id else 'Ninguna'}</p>
    <p><strong>Variables numéricas utilizadas en el modelo:</strong> {', '.join(cols_num)}</p>
    <p><strong>Resumen de imputación:</strong></p>
    <ul>{''.join(f'<li>{item}</li>' for item in resumen_imp)}</ul>
    <p><strong>Reglas de negocio aplicadas:</strong></p>
    <ul>{''.join(f'<li>{item}</li>' for item in resumen_reg)}</ul>
  </section>

  <section class="seccion">
    <h2>2. Metodología de Clustering y Métricas de Evaluación</h2>
    <div class="kpi-grid">
      <div class="kpi"><small>Algoritmo</small><strong>{st.session_state.algoritmo}</strong></div>
      <div class="kpi"><small>Número de clusters (K)</small><strong>{st.session_state.k_seleccionado}</strong></div>
      <div class="kpi"><small>Escalamiento</small><strong>{st.session_state.metodo_escalamiento}</strong></div>
      <div class="kpi"><small>Silhouette Score</small><strong>{sil_texto}</strong></div>
    </div>
    <p style="margin-top:14px;">
      <strong>Configuración de distancia:</strong> {st.session_state.metrica_distancia}
      {f" · Enlace: {st.session_state.metodo_enlace}" if st.session_state.algoritmo != "K-Means" else ""}
      {f" · Nota de negocio: {st.session_state.criterio_negocio_nota}" if st.session_state.criterio_negocio_nota else ""}
    </p>
  </section>

  <section class="seccion">
    <h2>3. Tabla de Centroides Desescalados (Unidades Originales) y Tamaño por Segmento</h2>
    {tabla_centroides_html}
  </section>

  <section class="seccion">
    <h2>4. Visualización PCA 2D y Gráfico de Radar</h2>
    <p>El análisis de componentes principales (PCA) se utiliza exclusivamente para proyectar los segmentos en dos dimensiones sin sustituir las variables originales del modelo.</p>
    <div>{html_pca}</div>
    <div style="margin-top:20px;">{html_radar}</div>
  </section>

  <section class="seccion">
    <h2>5. Perfiles Comerciales y Estrategias por Segmento</h2>
    <p>Modalidad de generación: <strong>{st.session_state.modo_ia_usado}</strong>. Las recomendaciones se fundamentan en las características estadísticas observadas en los centroides.</p>
    {''.join(tarjetas_segmentos)}
  </section>

  <section class="seccion">
    <h2>6. Conclusiones Ejecutivas</h2>
    <p>
      El proceso de segmentación sobre <strong>{df_res.shape[0]:,}</strong> registros válidos del sector
      <strong>{sector}</strong> permitió consolidar <strong>{st.session_state.k_seleccionado}</strong> grupos
      diferenciados a partir de {len(cols_num)} variables numéricas. El segmento de mayor representación es
      <strong>{nombre_mayor}</strong> con {int(segmento_mayor['N']):,} registros ({float(segmento_mayor['Porcentaje']):.1f}%).
    </p>
    <p>
      Se recomienda implementar planes piloto diferenciados por cluster priorizando las acciones de Producto,
      Precio, Innovación, Comunicación y Rentabilidad descritas en este documento, evaluando su evolución con
      mediciones periódicas antes de extrapolar relaciones causales.
    </p>
  </section>
</div>
</body>
</html>
"""
    return html


# ==============================================================================
# RENDERIZADO DE BARRA LATERAL Y NAVEGACIÓN
# ==============================================================================
def renderizar_sidebar() -> Dict[str, str]:
    """Configura la barra lateral con progreso de 13 pasos y credenciales seguras."""
    with st.sidebar:
        st.markdown("### Clustering & Commercial Strategy Lab")
        st.text_input(
            "Nombre del proyecto",
            key="nombre_proyecto",
            help="Se mostrará en la portada del informe ejecutivo HTML.",
        )

        st.markdown("---")
        st.markdown("#### Configuración OpenAI (Opcional)")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="La clave nunca se almacena en el código ni en los informes exportados.",
        )
        st.selectbox(
            "Modelo OpenAI",
            options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
            key="openai_model",
        )
        if api_key.strip():
            st.success("Modo IA OpenAI habilitado.")
        else:
            st.info("Sin API Key: operará en Modo Simulación / Plantilla.")

        st.markdown("---")
        paso_actual = st.session_state.paso_actual
        st.progress(paso_actual / 13.0, text=f"Progreso del flujo: Paso {paso_actual} de 13")

        st.markdown("#### Etapas del análisis")
        for num_paso, titulo in PASOS_FLUJO:
            habilitado = paso_disponible(num_paso)
            tipo = "primary" if num_paso == paso_actual else "secondary"
            if st.button(
                titulo,
                key=f"nav_paso_{num_paso}",
                use_container_width=True,
                disabled=not habilitado,
                type=tipo,
            ):
                st.session_state.paso_actual = num_paso
                st.rerun()

    return {"api_key": api_key.strip(), "modelo": st.session_state.openai_model}


def renderizar_navegacion_inferior(paso: int, puede_avanzar: bool = True) -> None:
    """Renderiza botones secuenciales Atrás / Siguiente conservando el estado."""
    st.markdown("---")
    col_izq, col_centro, col_der = st.columns([1, 2, 1])
    with col_izq:
        if paso > 1:
            if st.button("← Paso anterior", use_container_width=True, key=f"prev_{paso}"):
                st.session_state.paso_actual = paso - 1
                st.rerun()
    with col_der:
        if paso < 13:
            if st.button(
                "Siguiente paso →",
                type="primary",
                use_container_width=True,
                disabled=not puede_avanzar,
                key=f"next_{paso}",
            ):
                st.session_state.paso_actual = paso + 1
                st.rerun()


# ==============================================================================
# VISTAS DE LAS 13 FUNCIONALIDADES
# ==============================================================================
def vista_paso_1() -> None:
    st.subheader("3.1. Carga de archivos multiformato")
    st.write("Importe su base de datos en formato **CSV, TXT, XLS o XLSX**, o cargue la base de demostración.")

    col1, col2 = st.columns(2)
    with col1:
        separador_label = st.selectbox(
            "Separador (para CSV / TXT)",
            options=["Coma (,)", "Punto y coma (;)", "Tabulador (\\t)"],
        )
        mapa_sep = {"Coma (,)": ",", "Punto y coma (;)": ";", "Tabulador (\\t)": "\t"}
        separador = mapa_sep[separador_label]
    with col2:
        codificacion = st.selectbox(
            "Codificación de caracteres",
            options=["utf-8", "latin-1"],
        )

    archivo = st.file_uploader(
        "Seleccione el archivo de datos",
        type=["csv", "txt", "xls", "xlsx"],
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if archivo is not None and st.button("Procesar archivo cargado", type="primary", use_container_width=True):
            try:
                df = cargar_archivo_multiformato(archivo, separador, codificacion)
                invalidar_desde_paso(1)
                st.session_state.df_original = df
                st.session_state.df_trabajo = df.copy()
                st.session_state.nombre_archivo = archivo.name
                st.success(f"Archivo '{archivo.name}' cargado correctamente: {df.shape[0]:,} registros y {df.shape[1]} columnas.")
            except Exception as exc:
                st.error(f"Error al cargar el archivo: {exc}")

    with col_btn2:
        if st.button("Cargar base de datos empresarial de ejemplo", use_container_width=True):
            df_ej = generar_dataset_ejemplo()
            invalidar_desde_paso(1)
            st.session_state.df_original = df_ej
            st.session_state.df_trabajo = df_ej.copy()
            st.session_state.nombre_archivo = "clientes_ejemplo_retail.csv"
            st.success("Base empresarial de ejemplo cargada (183 registros con nulos, duplicados y atípicos de prueba).")

    if st.session_state.df_original is not None:
        df = st.session_state.df_original
        st.info(f"Base activa en memoria: **{st.session_state.nombre_archivo}** ({df.shape[0]:,} filas × {df.shape[1]} columnas).")

    renderizar_navegacion_inferior(1, puede_avanzar=st.session_state.df_original is not None)


def vista_paso_2() -> None:
    st.subheader("3.2. Vista previa y selección del sector económico")
    df = st.session_state.df_original

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total de registros</div><div class="kpi-value">{df.shape[0]:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total de columnas</div><div class="kpi-value">{df.shape[1]}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Dimensiones generales</div><div class="kpi-value">{df.shape[0]:,} × {df.shape[1]}</div></div>', unsafe_allow_html=True)

    st.markdown("#### Primeras cinco filas (`df.head(5)`)")
    st.dataframe(df.head(5), use_container_width=True)

    st.markdown("#### Sector económico (obligatorio)")
    sector = st.selectbox(
        "Seleccione el sector económico de la base de datos",
        options=SECTORES_ECONOMICOS,
        index=SECTORES_ECONOMICOS.index(st.session_state.sector_economico)
        if st.session_state.sector_economico in SECTORES_ECONOMICOS
        else 0,
    )
    st.session_state.sector_economico = sector

    sector_valido = True
    if sector == "Otro":
        especifico = st.text_input(
            "Especifique el sector económico",
            value=st.session_state.sector_personalizado,
            placeholder="Ej. Turismo y hotelería, Agroindustria...",
        )
        st.session_state.sector_personalizado = especifico
        if not especifico.strip():
            st.warning("Debe especificar el nombre del sector económico cuando selecciona 'Otro'.")
            sector_valido = False

    renderizar_navegacion_inferior(2, puede_avanzar=sector_valido)


def vista_paso_3() -> None:
    st.subheader("3.3. Inspección y clasificación de variables")
    df = st.session_state.df_original
    tabla_inspeccion = construir_tabla_inspeccion(df)
    st.dataframe(tabla_inspeccion, use_container_width=True, hide_index=True)

    todas_cols = df.columns.tolist()
    cols_id_por_defecto = [
        c for c in st.session_state.cols_id if c in todas_cols
    ] or [c for c in todas_cols if "id" in str(c).lower()]

    cols_id = st.multiselect(
        "1. Seleccione variables identificadoras (IDs) a excluir del modelo",
        options=todas_cols,
        default=cols_id_por_defecto,
        help="Estas columnas se conservarán para trazabilidad pero nunca ingresarán al modelo.",
    )

    candidatas_numericas = [
        c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c not in cols_id
    ]

    if not candidatas_numericas:
        st.error("No hay columnas numéricas disponibles fuera de los identificadores seleccionados.")
        renderizar_navegacion_inferior(3, puede_avanzar=False)
        return

    default_num = [c for c in st.session_state.cols_numericas if c in candidatas_numericas]
    if not default_num:
        default_num = candidatas_numericas

    cols_numericas = st.multiselect(
        "2. Seleccione las variables numéricas para el clustering (mínimo 2)",
        options=candidatas_numericas,
        default=default_num,
    )

    # Validar columnas sin variabilidad
    sin_varianza = [c for c in cols_numericas if df[c].nunique(dropna=True) <= 1]
    if sin_varianza:
        st.warning(f"Las siguientes columnas no tienen variabilidad y deben excluirse: {', '.join(sin_varianza)}")

    valido = len(cols_numericas) >= 2 and len(sin_varianza) == 0
    if len(cols_numericas) < 2:
        st.error("Debe seleccionar al menos **dos (2)** variables numéricas con variabilidad.")

    if set(cols_id) != set(st.session_state.cols_id) or set(cols_numericas) != set(st.session_state.cols_numericas):
        st.session_state.cols_id = cols_id
        st.session_state.cols_numericas = cols_numericas
        invalidar_desde_paso(3)

    renderizar_navegacion_inferior(3, puede_avanzar=valido)


def vista_paso_4() -> None:
    st.subheader("3.4. Limpieza y diagnóstico de datos faltantes")
    if st.session_state.df_limpio is None:
        st.session_state.df_limpio = st.session_state.df_original.copy()

    df_actual = st.session_state.df_limpio
    total_inicial = len(st.session_state.df_original)

    nulos = df_actual.isna().sum()
    pct_nulos = (nulos / max(len(df_actual), 1) * 100).round(2)
    diag_df = pd.DataFrame({
        "Variable": df_actual.columns,
        "Valores nulos": nulos.values,
        "Porcentaje faltante (%)": pct_nulos.values,
    }).sort_values("Valores nulos", ascending=False)

    duplicados_actuales = int(df_actual.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Registros iniciales</div><div class="kpi-value">{total_inicial:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Registros actuales</div><div class="kpi-value">{len(df_actual):,}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Duplicados detectados</div><div class="kpi-value">{duplicados_actuales:,}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Duplicados eliminados</div><div class="kpi-value">{st.session_state.duplicados_eliminados:,}</div></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Diagnóstico de nulos", "Registros duplicados"])
    with tab1:
        st.dataframe(diag_df, use_container_width=True, hide_index=True)
        fig_nulos = px.bar(
            diag_df,
            x="Variable",
            y="Porcentaje faltante (%)",
            text="Valores nulos",
            title="Porcentaje de valores faltantes por variable",
            color_discrete_sequence=[COLORES_CORPORATIVOS["azul_rey"]],
        )
        fig_nulos.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig_nulos, use_container_width=True)

    with tab2:
        if duplicados_actuales > 0:
            st.warning(f"Se detectaron **{duplicados_actuales}** registros duplicados exactos.")
            if st.button("Eliminar registros duplicados", type="primary"):
                antes = len(df_actual)
                df_sin_dup = df_actual.drop_duplicates().reset_index(drop=True)
                eliminados = antes - len(df_sin_dup)
                st.session_state.df_limpio = df_sin_dup
                st.session_state.duplicados_eliminados += eliminados
                invalidar_desde_paso(4)
                st.success(f"Se eliminaron {eliminados} duplicados. Registros restantes: {len(df_sin_dup):,}.")
                st.rerun()
        else:
            st.success("No existen registros duplicados en la base actual.")

    renderizar_navegacion_inferior(4, puede_avanzar=len(df_actual) >= 10)


def vista_paso_5() -> None:
    st.subheader("3.5. Imputación de datos")
    df_base = st.session_state.df_limpio if st.session_state.df_limpio is not None else st.session_state.df_original
    cols_con_nulos = [c for c in df_base.columns if df_base[c].isna().any()]

    if not cols_con_nulos:
        st.success("La base de datos no presenta valores faltantes.")
        if st.session_state.df_imputado is None:
            st.session_state.df_imputado = df_base.copy()
            st.session_state.resumen_imputacion = ["No se detectaron valores faltantes."]
        renderizar_navegacion_inferior(5, puede_avanzar=True)
        return

    st.write(f"Se detectaron **{len(cols_con_nulos)}** columnas con valores faltantes. La base original permanecerá intacta.")
    modo_global = st.radio(
        "Modo de tratamiento",
        options=["Configurar estrategia de imputación por columna", "Eliminar registros que contengan valores faltantes"],
    )

    estrategias_col = {}
    if modo_global == "Configurar estrategia de imputación por columna":
        for col in cols_con_nulos:
            es_num = pd.api.types.is_numeric_dtype(df_base[col])
            n_miss = int(df_base[col].isna().sum())
            opciones = ["Media", "Mediana", "Cero", "KNNImputer"] if es_num else ["Moda", 'Sustitución por "Desconocido"']
            estrategias_col[col] = st.selectbox(
                f"{col} ({'Numérica' if es_num else 'Categórica'} · {n_miss} nulos)",
                options=opciones,
                key=f"imp_{col}",
            )

    if st.button("Aplicar tratamiento de valores faltantes", type="primary"):
        df_proc = df_base.copy()
        resumen = []

        if modo_global == "Eliminar registros que contengan valores faltantes":
            antes = len(df_proc)
            df_proc = df_proc.dropna().reset_index(drop=True)
            resumen.append(f"Se eliminaron {antes - len(df_proc)} registros con valores faltantes (quedan {len(df_proc):,}).")
        else:
            # Aplicar KNNImputer primero si alguna numérica lo requiere
            cols_knn = [c for c, est in estrategias_col.items() if est == "KNNImputer"]
            if cols_knn:
                todas_num = df_proc.select_dtypes(include=[np.number]).columns.tolist()
                imputador = KNNImputer(n_neighbors=min(5, max(1, len(df_proc) - 1)))
                matriz_knn = pd.DataFrame(
                    imputador.fit_transform(df_proc[todas_num]),
                    columns=todas_num,
                    index=df_proc.index,
                )
                for col in cols_knn:
                    n_miss = int(df_proc[col].isna().sum())
                    df_proc[col] = matriz_knn[col]
                    resumen.append(f"{col}: {n_miss} nulos imputados mediante KNNImputer.")

            for col, est in estrategias_col.items():
                if est == "KNNImputer":
                    continue
                n_miss = int(df_proc[col].isna().sum())
                if est == "Media":
                    val = float(df_proc[col].mean())
                    df_proc[col] = df_proc[col].fillna(val)
                    resumen.append(f"{col}: {n_miss} nulos imputados con Media ({val:,.2f}).")
                elif est == "Mediana":
                    val = float(df_proc[col].median())
                    df_proc[col] = df_proc[col].fillna(val)
                    resumen.append(f"{col}: {n_miss} nulos imputados con Mediana ({val:,.2f}).")
                elif est == "Cero":
                    df_proc[col] = df_proc[col].fillna(0)
                    resumen.append(f"{col}: {n_miss} nulos imputados con 0.")
                elif est == "Moda":
                    moda = df_proc[col].mode(dropna=True)
                    val = moda.iloc[0] if not moda.empty else "Desconocido"
                    df_proc[col] = df_proc[col].fillna(val)
                    resumen.append(f"{col}: {n_miss} nulos imputados con Moda ('{val}').")
                else:
                    df_proc[col] = df_proc[col].fillna("Desconocido")
                    resumen.append(f"{col}: {n_miss} nulos sustituidos por 'Desconocido'.")

        st.session_state.df_imputado = df_proc
        st.session_state.resumen_imputacion = resumen
        invalidar_desde_paso(5)
        st.success("Imputación aplicada sin alterar la base original.")

    if st.session_state.resumen_imputacion:
        st.markdown("#### Resumen de cambios realizados")
        for item in st.session_state.resumen_imputacion:
            st.write(f"- {item}")

    puede_avanzar = (
        st.session_state.df_imputado is not None
        and not st.session_state.df_imputado[st.session_state.cols_numericas].isna().any().any()
    )
    renderizar_navegacion_inferior(5, puede_avanzar=puede_avanzar)


def vista_paso_6() -> None:
    st.subheader("3.6. Análisis descriptivo y reglas de negocio")
    df_imp = st.session_state.df_imputado if st.session_state.df_imputado is not None else st.session_state.df_limpio
    cols_num = st.session_state.cols_numericas

    tab_desc, tab_graf, tab_reglas = st.tabs([
        "Estadísticas descriptivas",
        "Histogramas y Boxplots",
        "Reglas de negocio y límites",
    ])

    with tab_desc:
        st.dataframe(df_imp[cols_num].describe().T, use_container_width=True)

    with tab_graf:
        var_sel = st.selectbox("Seleccione variable para explorar distribución y atípicos", options=cols_num)
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(
                df_imp,
                x=var_sel,
                nbins=30,
                title=f"Histograma: {var_sel}",
                color_discrete_sequence=[COLORES_CORPORATIVOS["azul_rey"]],
            )
            fig_hist.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig_hist, use_container_width=True)
        with c2:
            fig_box = px.box(
                df_imp,
                y=var_sel,
                points="outliers",
                title=f"Diagrama de caja y atípicos: {var_sel}",
                color_discrete_sequence=[COLORES_CORPORATIVOS["azul_aguamarina"]],
            )
            fig_box.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig_box, use_container_width=True)

    with tab_reglas:
        st.write("Defina límites mínimos y máximos permitidos para excluir registros fuera de rango de negocio.")
        limites = {}
        for col in cols_num:
            min_real = float(df_imp[col].min())
            max_real = float(df_imp[col].max())
            col_min, col_max = st.columns(2)
            with col_min:
                val_min = st.number_input(
                    f"Mínimo permitido ({col})",
                    value=st.session_state.reglas_negocio.get(col, (min_real, max_real))[0],
                    key=f"min_{col}",
                )
            with col_max:
                val_max = st.number_input(
                    f"Máximo permitido ({col})",
                    value=st.session_state.reglas_negocio.get(col, (min_real, max_real))[1],
                    key=f"max_{col}",
                )
            limites[col] = (val_min, val_max)

        if st.button("Aplicar reglas de negocio", type="primary"):
            mask_global = pd.Series(True, index=df_imp.index)
            resumen_reglas = []
            for col, (v_min, v_max) in limites.items():
                if v_min > v_max:
                    st.error(f"En '{col}', el mínimo ({v_min}) no puede ser mayor al máximo ({v_max}).")
                    return
                incumplen = ~df_imp[col].between(v_min, v_max)
                n_excl = int(incumplen.sum())
                mask_global &= ~incumplen
                if n_excl > 0:
                    resumen_reglas.append(f"{col} fuera de [{v_min:,.2f}, {v_max:,.2f}]: {n_excl} registros excluidos.")
            df_filt = df_imp[mask_global].reset_index(drop=True)
            if len(df_filt) < 4:
                st.error("Las reglas definidas dejan menos de 4 registros disponibles. Ajuste los rangos.")
                return
            st.session_state.reglas_negocio = limites
            st.session_state.resumen_reglas = resumen_reglas or ["Todos los registros cumplen los rangos definidos."]
            st.session_state.df_filtrado = df_filt
            invalidar_desde_paso(6)
            st.success(f"Reglas aplicadas. Total disponible para modelar: {len(df_filt):,} registros.")

    if st.session_state.df_filtrado is None:
        st.session_state.df_filtrado = df_imp.copy()

    df_disp = st.session_state.df_filtrado
    st.info(f"Registros disponibles para modelar: **{len(df_disp):,}** (de {len(df_imp):,} tras imputación).")
    if st.session_state.resumen_reglas:
        for r in st.session_state.resumen_reglas:
            st.write(f"- {r}")

    renderizar_navegacion_inferior(6, puede_avanzar=len(df_disp) >= 4)


def vista_paso_7() -> None:
    st.subheader("3.7. Determinación del número óptimo de clusters (K)")
    df_modelo = obtener_df_para_modelar()
    cols_num = st.session_state.cols_numericas

    metodo_esc = st.selectbox(
        "Método de escalamiento automático",
        options=["StandardScaler", "MinMaxScaler"],
        index=0 if st.session_state.metodo_escalamiento == "StandardScaler" else 1,
    )
    if metodo_esc != st.session_state.metodo_escalamiento:
        st.session_state.metodo_escalamiento = metodo_esc
        st.session_state.resultados_k = None
        invalidar_desde_paso(8)

    if st.button("Calcular métricas para K = 2 a 10", type="primary") or st.session_state.resultados_k is None:
        try:
            matriz_esc, _ = calcular_matriz_escalada(df_modelo, cols_num, metodo_esc)
            df_k = evaluar_rango_k(matriz_esc, k_max=10)
            st.session_state.resultados_k = df_k
        except Exception as exc:
            st.error(f"Error al evaluar rango de K: {exc}")
            return

    df_k = st.session_state.resultados_k
    mejor_sil_k, codo_k = recomendar_k_tecnico(df_k)

    c1, c2 = st.columns(2)
    with c1:
        fig_codo = px.line(
            df_k,
            x="K",
            y="Inercia",
            markers=True,
            title=f"A. Método del Codo (Inflexión sugerida: K={codo_k})",
            color_discrete_sequence=[COLORES_CORPORATIVOS["azul_rey"]],
        )
        fig_codo.add_vline(x=codo_k, line_dash="dash", line_color=COLORES_CORPORATIVOS["verde_biche"])
        fig_codo.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig_codo, use_container_width=True)

    with c2:
        fig_sil = px.line(
            df_k,
            x="K",
            y="Silhouette",
            markers=True,
            title=f"B. Coeficiente de Silueta (Mejor técnico: K={mejor_sil_k})",
            color_discrete_sequence=[COLORES_CORPORATIVOS["azul_aguamarina"]],
        )
        fig_sil.add_vline(x=mejor_sil_k, line_dash="dash", line_color=COLORES_CORPORATIVOS["amarillo"])
        fig_sil.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig_sil, use_container_width=True)

    st.dataframe(df_k, use_container_width=True, hide_index=True)

    st.markdown("#### C. Criterio de negocio y decisión final")
    st.info(
        f"Recomendación técnica: Silueta máxima en **K = {mejor_sil_k}** e inflexión del codo en **K = {codo_k}**. "
        "Evalúe la capacidad operativa, recursos comerciales disponibles y facilidad de gestión antes de fijar el K definitivo."
    )

    k_max_valido = int(df_k["K"].max())
    k_elegido = st.slider(
        "Número final de segmentos a construir (K)",
        min_value=2,
        max_value=k_max_valido,
        value=min(max(2, st.session_state.k_seleccionado), k_max_valido),
    )
    nota_neg = st.text_input(
        "Justificación operativa / comercial (opcional)",
        value=st.session_state.criterio_negocio_nota,
        placeholder="Ej. Capacidad del equipo comercial para gestionar 4 estrategias diferenciadas.",
    )

    if k_elegido != st.session_state.k_seleccionado:
        st.session_state.k_seleccionado = k_elegido
        invalidar_desde_paso(8)
    st.session_state.criterio_negocio_nota = nota_neg

    renderizar_navegacion_inferior(7, puede_avanzar=True)


def vista_paso_8() -> None:
    st.subheader("3.8. Selección del algoritmo y métrica de distancia")
    algo = st.radio(
        "Seleccione el algoritmo de segmentación",
        options=["K-Means", "Clustering jerárquico o aglomerativo"],
        index=0 if st.session_state.algoritmo == "K-Means" else 1,
    )
    st.session_state.algoritmo = algo

    valido = True
    if algo == "K-Means":
        st.info("K-Means opera exclusivamente con distancia **Euclidiana**.")
        st.session_state.metrica_distancia = "euclidean"
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.kmeans_n_init = st.number_input("Número de inicializaciones (n_init)", min_value=1, max_value=50, value=int(st.session_state.kmeans_n_init))
            st.session_state.kmeans_random_state = st.number_input("Semilla aleatoria (random_state)", min_value=0, max_value=9999, value=int(st.session_state.kmeans_random_state))
        with c2:
            st.session_state.kmeans_max_iter = st.number_input("Iteraciones máximas (max_iter)", min_value=50, max_value=1000, value=int(st.session_state.kmeans_max_iter))
            st.session_state.kmeans_tol = st.number_input("Tolerancia de convergencia (tol)", min_value=1e-6, max_value=1e-1, value=float(st.session_state.kmeans_tol), format="%.6f")
    else:
        c1, c2 = st.columns(2)
        with c1:
            metrica = st.selectbox(
                "Métrica de distancia",
                options=["euclidean", "manhattan", "cosine"],
                format_func=lambda x: {"euclidean": "Euclidiana", "manhattan": "Manhattan", "cosine": "Coseno"}[x],
            )
        with c2:
            enlace = st.selectbox(
                "Método de enlace (linkage)",
                options=["ward", "complete", "average", "single"],
                format_func=lambda x: x.capitalize(),
            )
        st.session_state.metrica_distancia = metrica
        st.session_state.metodo_enlace = enlace

        valido, msg = validar_jerarquico(metrica, enlace)
        if valido:
            st.success(f"{msg} ({metrica} + {enlace})")
        else:
            st.error(msg)

    renderizar_navegacion_inferior(8, puede_avanzar=valido)


def vista_paso_9() -> None:
    st.subheader("3.9. Ejecución del clustering")
    df_modelo = obtener_df_para_modelar()
    cols_num = st.session_state.cols_numericas
    k = st.session_state.k_seleccionado

    if len(df_modelo) <= k:
        st.error(f"Cantidad insuficiente de registros ({len(df_modelo)}) para {k} clusters.")
        renderizar_navegacion_inferior(9, puede_avanzar=False)
        return

    if st.session_state.algoritmo != "K-Means":
        valido, msg = validar_jerarquico(st.session_state.metrica_distancia, st.session_state.metodo_enlace)
        if not valido:
            st.error(msg)
            renderizar_navegacion_inferior(9, puede_avanzar=False)
            return

    if st.button("Ejecutar modelo de clustering", type="primary"):
        try:
            matriz_esc, scaler = calcular_matriz_escalada(df_modelo, cols_num, st.session_state.metodo_escalamiento)

            if st.session_state.algoritmo == "K-Means":
                modelo = KMeans(
                    n_clusters=k,
                    n_init=int(st.session_state.kmeans_n_init),
                    random_state=int(st.session_state.kmeans_random_state),
                    max_iter=int(st.session_state.kmeans_max_iter),
                    tol=float(st.session_state.kmeans_tol),
                )
                etiquetas = modelo.fit_predict(matriz_esc)
                inercia = float(modelo.inertia_)
            else:
                modelo = AgglomerativeClustering(
                    n_clusters=k,
                    metric=st.session_state.metrica_distancia,
                    linkage=st.session_state.metodo_enlace,
                )
                etiquetas = modelo.fit_predict(matriz_esc)
                inercia = None

            df_res = df_modelo.copy()
            df_res["Cluster"] = etiquetas

            # Calcular centroides desescalados como las medias reales sobre las unidades originales
            centroides = df_res.groupby("Cluster")[cols_num].mean().reset_index()
            conteos = df_res["Cluster"].value_counts().sort_index()
            centroides["N"] = centroides["Cluster"].map(conteos)
            centroides["Porcentaje"] = (centroides["N"] / len(df_res) * 100).round(2)

            # PCA 2D únicamente para visualización
            pca = PCA(n_components=2, random_state=42)
            pca_coords = pca.fit_transform(matriz_esc)

            metricas = {
                "silhouette": float(silhouette_score(matriz_esc, etiquetas)),
                "calinski_harabasz": float(calinski_harabasz_score(matriz_esc, etiquetas)),
                "davies_bouldin": float(davies_bouldin_score(matriz_esc, etiquetas)),
                "inercia": inercia,
            }

            st.session_state.df_resultado = df_res
            st.session_state.centroides_df = centroides
            st.session_state.pca_coords = pca_coords
            st.session_state.pca_varianza = pca.explained_variance_ratio_
            st.session_state.pca_modelo = pca
            st.session_state.scaler_modelo = scaler
            st.session_state.metricas_modelo = metricas
            invalidar_desde_paso(10)
            st.success("Segmentación ejecutada exitosamente.")
        except Exception as exc:
            st.error(f"Error durante la ejecución del modelo: {exc}")

    if st.session_state.df_resultado is not None:
        df_res = st.session_state.df_resultado
        met = st.session_state.metricas_modelo
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Segmentos generados</div><div class="kpi-value">{df_res["Cluster"].nunique()}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Registros clasificados</div><div class="kpi-value">{len(df_res):,}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Silhouette Score</div><div class="kpi-value">{met["silhouette"]:.4f}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Davies-Bouldin</div><div class="kpi-value">{met["davies_bouldin"]:.4f}</div></div>', unsafe_allow_html=True)

        st.markdown("#### Muestra de registros clasificados con etiqueta `Cluster`")
        st.dataframe(df_res.head(20), use_container_width=True)

    renderizar_navegacion_inferior(9, puede_avanzar=st.session_state.df_resultado is not None)


def vista_paso_10() -> None:
    st.subheader("3.10. Visualización de centroides y perfiles")
    centroides_df = st.session_state.centroides_df
    cols_num = st.session_state.cols_numericas
    df_res = st.session_state.df_resultado
    perfiles = st.session_state.perfiles_clusters

    st.markdown("#### Tabla de centroides desescalados (unidades originales)")
    st.dataframe(
        centroides_df.style.format({c: "{:,.2f}" for c in cols_num + ["Porcentaje"]}),
        use_container_width=True,
        hide_index=True,
    )

    tab_pca, tab_radar = st.tabs(["Gráfico PCA 2D", "Gráfico de Radar"])
    with tab_pca:
        fig_pca = crear_figura_pca(
            st.session_state.pca_coords,
            df_res["Cluster"].values,
            st.session_state.pca_varianza,
            centroides_df,
            cols_num,
            st.session_state.pca_modelo,
            st.session_state.scaler_modelo,
            perfiles,
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with tab_radar:
        fig_radar = crear_figura_radar(centroides_df, cols_num, perfiles)
        st.plotly_chart(fig_radar, use_container_width=True)

    renderizar_navegacion_inferior(10, puede_avanzar=True)


def vista_paso_11(config_ia: Dict[str, str]) -> None:
    st.subheader("3.11. Perfilamiento automático y nombres comerciales")
    sector = obtener_sector_efectivo()
    centroides_df = st.session_state.centroides_df
    cols_num = st.session_state.cols_numericas
    df_res = st.session_state.df_resultado

    api_key = config_ia["api_key"]
    modelo = config_ia["modelo"]

    if st.button("Generar perfiles y nombres comerciales", type="primary"):
        if api_key:
            try:
                with st.spinner("Consultando OpenAI API para perfilamiento comercial..."):
                    perfiles, estrategias = generar_perfiles_openai(
                        api_key, modelo, centroides_df, cols_num, sector
                    )
                st.session_state.perfiles_clusters = perfiles
                st.session_state.estrategias_clusters = estrategias
                st.session_state.modo_ia_usado = f"Inteligencia Artificial ({modelo})"
                st.success("Perfiles y estrategias generados mediante OpenAI API.")
            except Exception as exc:
                st.warning(f"No fue posible completar la llamada a OpenAI ({exc}). Se activará el Modo Simulación / Plantilla.")
                perfiles, estrategias = generar_perfiles_simulacion(centroides_df, cols_num, df_res, sector)
                st.session_state.perfiles_clusters = perfiles
                st.session_state.estrategias_clusters = estrategias
                st.session_state.modo_ia_usado = "Simulación / Plantilla (respaldo)"
        else:
            perfiles, estrategias = generar_perfiles_simulacion(centroides_df, cols_num, df_res, sector)
            st.session_state.perfiles_clusters = perfiles
            st.session_state.estrategias_clusters = estrategias
            st.session_state.modo_ia_usado = "Simulación / Plantilla"
            st.success("Perfiles generados mediante el motor estructurado de simulación.")

    if st.session_state.perfiles_clusters:
        st.caption(f"Modalidad activa: **{st.session_state.modo_ia_usado}** · Sector: **{sector}**")
        for _, fila in centroides_df.iterrows():
            cid = int(fila["Cluster"])
            p = st.session_state.perfiles_clusters[cid]
            with st.expander(f"Cluster {cid} — {p['nombre_comercial']} ({int(fila['N'])} registros · {fila['Porcentaje']:.1f}%)", expanded=True):
                st.write(f"**Descripción del perfil:** {p['descripcion_perfil']}")
                st.write(f"**Comportamiento predominante:** {p['comportamiento_predominante']}")
                st.write(f"**Necesidades relevantes:** {p['necesidades_relevantes']}")
                st.write(f"**Oportunidades comerciales:** {p['oportunidades_comerciales']}")
                st.write("**Características diferenciadoras:**")
                for c_dif in p["caracteristicas_diferenciadoras"]:
                    st.write(f"- {c_dif}")

    renderizar_navegacion_inferior(11, puede_avanzar=st.session_state.perfiles_clusters is not None)


def vista_paso_12() -> None:
    st.subheader("3.12. Generación de estrategias comerciales por cluster")
    estrategias = st.session_state.estrategias_clusters
    perfiles = st.session_state.perfiles_clusters
    centroides_df = st.session_state.centroides_df

    tabs = st.tabs([
        f"C{int(f['Cluster'])}: {perfiles[int(f['Cluster'])]['nombre_comercial'][:24]}"
        for _, f in centroides_df.iterrows()
    ])

    for tab, (_, fila) in zip(tabs, centroides_df.iterrows()):
        cid = int(fila["Cluster"])
        est = estrategias[cid]
        with tab:
            st.markdown(f"#### {perfiles[cid]['nombre_comercial']} ({int(fila['N'])} registros · {fila['Porcentaje']:.1f}%)")
            for pilar in ["Producto", "Precio", "Innovación", "Comunicación", "Rentabilidad"]:
                items_html = "".join(f"<li>{acc}</li>" for acc in est.get(pilar, []))
                st.markdown(
                    f"""
                    <div class="pilar-box">
                        <div class="pilar-titulo">{pilar}</div>
                        <ul>{items_html}</ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    renderizar_navegacion_inferior(12, puede_avanzar=True)


def vista_paso_13() -> None:
    st.subheader("3.13. Exportación del informe ejecutivo en HTML")
    st.write("Descargue un informe HTML autocontenido con tablas, gráficos interactivos Plotly, perfiles y estrategias de los cinco pilares.")

    centroides_df = st.session_state.centroides_df
    cols_num = st.session_state.cols_numericas
    df_res = st.session_state.df_resultado
    perfiles = st.session_state.perfiles_clusters

    fig_pca = crear_figura_pca(
        st.session_state.pca_coords,
        df_res["Cluster"].values,
        st.session_state.pca_varianza,
        centroides_df,
        cols_num,
        st.session_state.pca_modelo,
        st.session_state.scaler_modelo,
        perfiles,
    )
    fig_radar = crear_figura_radar(centroides_df, cols_num, perfiles)
    html_informe = generar_html_informe_ejecutivo(fig_pca, fig_radar)

    st.download_button(
        label="Descargar Informe Ejecutivo (.html)",
        data=html_informe.encode("utf-8"),
        file_name=f"informe_clustering_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
        mime="text/html",
        type="primary",
        use_container_width=True,
    )

    with st.expander("Vista previa resumida del documento"):
        st.write(f"- **Proyecto:** {st.session_state.nombre_proyecto}")
        st.write(f"- **Sector económico:** {obtener_sector_efectivo()}")
        st.write(f"- **Algoritmo y K:** {st.session_state.algoritmo} (K = {st.session_state.k_seleccionado})")
        st.write(f"- **Registros clasificados:** {len(df_res):,}")
        st.write(f"- **Tamaño aproximado del HTML autocontenido:** {len(html_informe) / 1024:,.1f} KB")

    renderizar_navegacion_inferior(13, puede_avanzar=False)


# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================
def main() -> None:
    st.set_page_config(
        page_title="Clustering & Commercial Strategy Lab",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inicializar_estado()
    inyectar_estilos()

    config_ia = renderizar_sidebar()

    st.markdown(
        """
        <div class="banner-institucional">
            <h1>Clustering &amp; Commercial Strategy Lab</h1>
            <p>Analítica de Datos · Segmentación Machine Learning · Inteligencia Comercial Accionable</p>
        </div>
        <div class="franja-colores">
            <div style="flex:1; background:#78BE20;"></div>
            <div style="flex:1; background:#FCD116;"></div>
            <div style="flex:1; background:#00B5AD;"></div>
            <div style="flex:1; background:#0033A0;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    paso = st.session_state.paso_actual
    rutas = {
        1: vista_paso_1,
        2: vista_paso_2,
        3: vista_paso_3,
        4: vista_paso_4,
        5: vista_paso_5,
        6: vista_paso_6,
        7: vista_paso_7,
        8: vista_paso_8,
        9: vista_paso_9,
        10: vista_paso_10,
        11: lambda: vista_paso_11(config_ia),
        12: vista_paso_12,
        13: vista_paso_13,
    }
    rutas.get(paso, vista_paso_1)()


if __name__ == "__main__":
    main()
