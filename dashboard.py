import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile

# Configuración
st.set_page_config(layout="wide", page_title="Dashboard de Indicadores")
st.title("📊 Análisis de Combinaciones de Indicadores Técnicos")

# Usar carpeta temporal compatible con Render
DATA_DIR = os.path.join(tempfile.gettempdir(), "data")

# Asegurarse de que exista
os.makedirs(DATA_DIR, exist_ok=True)

# Verificar archivos
try:
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
except Exception as e:
    st.error(f"❌ Error accediendo a {DATA_DIR}: {e}")
    st.stop()

if not csv_files:
    st.warning("⚠️ No hay archivos CSV. Ejecuta combinador.py o espera que GitHub los genere.")
    st.stop()


# Selector de archivo
file_selected = st.selectbox("📁 Selecciona un activo:", csv_files)
df = pd.read_csv(os.path.join(DATA_DIR, file_selected))

# Convertir a numérico las métricas, por si acaso vienen como string
for col in ["accuracy", "precision", "recall", "f1_score"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Filtro por F1 Score mínimo
min_f1 = st.slider("🎯 Filtrar combinaciones con F1 Score mínimo:", 0.0, 1.0, 0.7, 0.01)
filtered_df = df[df["f1_score"] >= min_f1]

st.write(f"Mostrando {len(filtered_df)} combinaciones con F1 Score ≥ {min_f1}")

# Mostrar tabla de resultados (ordenada)
st.dataframe(filtered_df.sort_values(by="f1_score", ascending=False).head(20), use_container_width=True)

# Gráfico de barras para las 10 mejores combinaciones
st.subheader("📈 Top 10 combinaciones por F1 Score")
top_10 = filtered_df.sort_values(by="f1_score", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=top_10, x="f1_score", y="indicadores", palette="viridis", ax=ax)
ax.set_title(f"Top combinaciones - {file_selected}", fontsize=14)
ax.set_xlabel("F1 Score")
ax.set_ylabel("Indicadores")
st.pyplot(fig)
