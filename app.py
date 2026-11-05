import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ─────────────────────────── Configuración ───────────────────────────
st.set_page_config(page_title="Billboard Top 1 Predictor", page_icon="🎵", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #f5f5f5;
}
h1, h2, h3 { color: #ffd166 !important; }
.hero {
    text-align:center; padding: 2rem 1rem; border-radius: 20px;
    background: linear-gradient(90deg, #ff006e, #8338ec, #3a86ff);
    box-shadow: 0 10px 30px rgba(0,0,0,.4); margin-bottom: 1.5rem;
}
.hero h1 { color: white !important; font-size: 2.6rem; margin: 0; }
.hero p { color: #f0f0f0; font-size: 1.1rem; margin-top: .5rem; }
.card {
    background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.15);
    border-radius: 16px; padding: 1.2rem; text-align: center;
}
.card .num { font-size: 2rem; font-weight: 800; color: #ffd166; }
.card .lbl { font-size: .9rem; color: #ddd; }
.result-yes {
    background: linear-gradient(90deg,#06d6a0,#118ab2); padding:1.5rem;
    border-radius:16px; text-align:center; font-size:1.6rem; font-weight:800; color:white;
}
.result-no {
    background: linear-gradient(90deg,#ef476f,#8338ec); padding:1.5rem;
    border-radius:16px; text-align:center; font-size:1.6rem; font-weight:800; color:white;
}
section[data-testid="stSidebar"] { background: #1b1838; }
</style>
""", unsafe_allow_html=True)

COLORS = ["#8338ec", "#ffd166"]
plt.rcParams.update({"axes.facecolor": "#1b1838", "figure.facecolor": "#1b1838",
                     "text.color": "white", "axes.labelcolor": "white",
                     "xtick.color": "white", "ytick.color": "white",
                     "axes.edgecolor": "#555"})

# ─────────────────────────── Datos ───────────────────────────
@st.cache_data
def cargar_datos(archivo):
    df = pd.read_csv(archivo)
    malas = df["title"].isna()
    if malas.any():
        rep = df.loc[malas, "id"].apply(lambda x: next(csv.reader([str(x)])))
        rep = pd.DataFrame(rep.tolist(), columns=df.columns)
        df = df.astype(object)
        df.loc[malas, df.columns] = rep.values
    for col in ["chart_date", "durationSeg", "top", "anioNacimiento"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["top"])
    df["top"] = df["top"].astype(int)
    return df

st.markdown("""
<div class="hero">
  <h1>🎤 ¿Llegará al #1 de Billboard? 🏆</h1>
  <p>Árbol de Decisión que predice si una canción alcanza el primer lugar</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("🎛️ Panel de control")
archivo = st.sidebar.file_uploader("Sube billboard.csv", type="csv")
if archivo is None:
    try:
        archivo = "billboard.csv"
        billboard = cargar_datos(archivo)
    except FileNotFoundError:
        st.info("📂 Sube el archivo **billboard.csv** en la barra lateral para comenzar.")
        st.stop()
else:
    billboard = cargar_datos(archivo)

st.sidebar.markdown("### 🌳 Hiperparámetros")
max_depth = st.sidebar.slider("Profundidad máxima", 1, 10, 4)
min_leaf = st.sidebar.slider("Mínimo de muestras por hoja", 1, 50, 10)
test_size = st.sidebar.slider("Tamaño de prueba", 0.1, 0.5, 0.3, 0.05)

# ─────────────────────────── Modelo ───────────────────────────
columnas = ["artist_type", "tempo", "genre", "durationSeg"]
X = pd.get_dummies(billboard[columnas].copy(), drop_first=True)
y = billboard["top"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=100, stratify=y)
arbol = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=30)
arbol.fit(X_train, y_train)
pred = arbol.predict(X_test)
acc = accuracy_score(y_test, pred)

# ─────────────────────────── Métricas ───────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, num, lbl in [
    (c1, len(billboard), "🎶 Canciones"),
    (c2, billboard["artist"].nunique(), "👩‍🎤 Artistas"),
    (c3, int(y.sum()), "🥇 Llegaron al #1"),
    (c4, f"{acc:.1%}", "🎯 Precisión"),
]:
    col.markdown(f'<div class="card"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
                 unsafe_allow_html=True)

st.write("")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploración", "🌳 Árbol", "📈 Evaluación", "🔮 Predictor"])

# ─────────────────────────── Exploración ───────────────────────────
with tab1:
    st.subheader("Vista de los datos")
    st.dataframe(billboard.head(15), use_container_width=True)

    var = st.selectbox("Variable a analizar", ["artist_type", "tempo", "genre"])
    a, b = st.columns(2)
    with a:
        st.markdown(f"**Proporción de Top 1 por `{var}`**")
        tabla = pd.crosstab(billboard[var], billboard["top"], normalize="index")
        fig, ax = plt.subplots(figsize=(6, 4))
        tabla.plot(kind="bar", ax=ax, color=COLORS)
        ax.legend(["No Top 1", "Sí Top 1"]); ax.set_ylabel("Proporción")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    with b:
        st.markdown(f"**Cantidad de canciones por `{var}`**")
        fig, ax = plt.subplots(figsize=(6, 4))
        billboard[var].value_counts().plot(kind="bar", ax=ax, color="#ff006e")
        ax.set_ylabel("Canciones")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    st.markdown("**Distribución de clases**")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(y.value_counts().sort_index(), labels=["No Top 1", "Sí Top 1"],
           colors=COLORS, autopct="%1.1f%%", textprops={"color": "white"})
    st.pyplot(fig, use_container_width=False)

# ─────────────────────────── Árbol ───────────────────────────
with tab2:
    st.subheader("Árbol de decisión entrenado")
    fig, ax = plt.subplots(figsize=(22, 10), facecolor="white")
    plot_tree(arbol, feature_names=X.columns, class_names=["No Top 1", "Sí Top 1"],
              filled=True, rounded=True, fontsize=9, ax=ax)
    st.pyplot(fig)

    st.subheader("Importancia de variables")
    imp = pd.DataFrame({"Variable": X.columns, "Importancia": arbol.feature_importances_})
    imp = imp[imp["Importancia"] > 0].sort_values("Importancia")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(imp["Variable"], imp["Importancia"], color="#3a86ff")
    st.pyplot(fig)

# ─────────────────────────── Evaluación ───────────────────────────
with tab3:
    a, b = st.columns(2)
    with a:
        st.subheader("Matriz de confusión")
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="magma")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=18,
                        color="white" if cm[i, j] < cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Top 1", "Sí Top 1"]); ax.set_yticklabels(["No Top 1", "Sí Top 1"])
        ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
        st.pyplot(fig)
    with b:
        st.subheader("Reporte de clasificación")
        rep = pd.DataFrame(classification_report(y_test, pred, output_dict=True)).T
        st.dataframe(rep.style.format("{:.2f}"), use_container_width=True)
        st.caption("⚠️ Las clases están desbalanceadas: el modelo acierta mucho en 'No Top 1' "
                   "pero le cuesta detectar los Top 1 (recall bajo en la clase 1).")

# ─────────────────────────── Predictor ───────────────────────────
with tab4:
    st.subheader("🎧 Crea tu canción y descubre su destino")
    a, b = st.columns(2)
    with a:
        tipo = st.selectbox("Tipo de artista", sorted(billboard["artist_type"].dropna().unique()))
        tempo = st.selectbox("Tempo", sorted(billboard["tempo"].dropna().unique()))
    with b:
        genero = st.selectbox("Género", sorted(billboard["genre"].dropna().unique()))
        dur = st.slider("Duración (segundos)", 60, 600, 210)
        st.caption(f"⏱️ {dur // 60}:{dur % 60:02d} minutos")

    if st.button("🔮 ¡Predecir!", use_container_width=True):
        nueva = pd.get_dummies(pd.DataFrame([{"durationSeg": dur, "artist_type": tipo,
                                              "tempo": tempo, "genre": genero}]))
        nueva = nueva.reindex(columns=X.columns, fill_value=False)
        res = arbol.predict(nueva)[0]
        prob = arbol.predict_proba(nueva)[0][1]
        if res == 1:
            st.balloons()
            st.markdown('<div class="result-yes">🏆 ¡Esta canción llegaría al TOP 1!</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-no">🎵 Esta canción no llegaría al Top 1</div>',
                        unsafe_allow_html=True)
        st.write("")
        st.progress(float(prob), text=f"Probabilidad de ser #1: {prob:.1%}")

st.markdown("---")
st.caption("Laboratorio de Árboles de Decisión · Hecho con Streamlit 🎶")
